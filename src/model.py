"""
Modelo de Programação Linear Inteira Mista (MILP) para roteirização e
alocação de equipes de campo com janelas de tempo suaves (soft time windows).

Formulação (resumo, ver README.md para a versão completa em notação
matemática):

Conjuntos
    N = {0, 1, ..., n}            nós: 0 = depósito, 1..n = pontos de atendimento
    K = {1, ..., k}                equipes disponíveis (frota homogênea, depósito único)

Variáveis de decisão
    x[i,j,k] ∈ {0,1}   equipe k percorre o arco (i -> j)
    t[i] >= 0           instante de início do atendimento no nó i (minutos desde 08:00)
    atraso[i] >= 0      atraso em relação ao prazo desejado (due) do nó i

Função objetivo
    min  soma( custo_min * tempo_viagem[i,j] * x[i,j,k] )
       + soma( peso_prioridade[i] * atraso[i] )

Restrições principais
    (1) cada ponto de atendimento é visitado exatamente uma vez (grau de entrada = 1)
    (2) cada ponto de atendimento é deixado exatamente uma vez (grau de saída = 1)
    (3) conservação de fluxo por equipe em cada nó visitado
    (4) cada equipe sai do depósito no máximo uma vez e retorna no máximo uma vez
    (5) propagação de tempo com eliminação de sub-rotas (estilo MTZ):
        t[j] >= t[i] + duracao[i] + tempo_viagem[i,j] - M*(1 - y[i,j])
    (6) atraso[i] >= t[i] - prazo[i],  atraso[i] >= 0
    (7) jornada máxima por equipe (tempo de viagem + atendimento) <= limite

O modelo é resolvido com o solver CBC (via PuLP), com limite de tempo
configurável — prática comum em problemas de roteirização reais, onde a
solução ótima comprovada nem sempre é alcançável em tempo hábil e reporta-se
o gap de otimalidade.
"""

import math
import time
from itertools import permutations

import numpy as np
import pandas as pd
import pulp

AVG_SPEED_KMH = 40.0
BIG_M = 3000.0


def euclidean_km(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def build_nodes(points: pd.DataFrame, depot_xy):
    """Retorna dict node_id -> (x, y), onde node 0 é o depósito."""
    nodes = {0: depot_xy}
    for _, row in points.iterrows():
        nodes[int(row["id"][1:])] = (row["x_km"], row["y_km"])
    return nodes


def build_travel_time_matrix(nodes: dict):
    ids = sorted(nodes.keys())
    dist = {}
    for i in ids:
        for j in ids:
            if i == j:
                continue
            d_km = euclidean_km(nodes[i], nodes[j])
            dist[(i, j)] = (d_km / AVG_SPEED_KMH) * 60.0  # minutos
    return dist


def solve_vrptw(points: pd.DataFrame, crews: pd.DataFrame, time_limit_sec: int = 120,
                 tardiness_weight_scale: float = 1.0, msg: bool = False):
    depot_xy = (crews.iloc[0]["depot_x_km"], crews.iloc[0]["depot_y_km"])
    nodes = build_nodes(points, depot_xy)
    customer_ids = [i for i in nodes if i != 0]
    all_ids = [0] + customer_ids
    K = list(crews["crew_id"])

    travel = build_travel_time_matrix(nodes)

    duration = {0: 0.0}
    due = {0: 0.0}
    earliest = {0: 0.0}
    priority = {0: 0}
    for _, row in points.iterrows():
        nid = int(row["id"][1:])
        duration[nid] = float(row["duration_min"])
        due[nid] = float(row["due_min"])
        earliest[nid] = float(row["earliest_start_min"])
        priority[nid] = int(row["priority"])

    priority_weight = {1: 8.0, 2: 3.0, 3: 1.0}
    max_minutes = dict(zip(crews["crew_id"], crews["max_minutes"]))

    prob = pulp.LpProblem("Roteirizacao_Equipes_MILP", pulp.LpMinimize)

    arcs = [(i, j) for i in all_ids for j in all_ids if i != j]
    x = pulp.LpVariable.dicts("x", (arcs, K), cat="Binary")
    t = pulp.LpVariable.dicts("t", customer_ids, lowBound=0, upBound=600)
    atraso = pulp.LpVariable.dicts("atraso", customer_ids, lowBound=0)

    # y[i,j] = uso do arco por qualquer equipe (auxiliar para restrições de tempo)
    y = {(i, j): pulp.lpSum(x[(i, j)][k] for k in K) for (i, j) in arcs}

    # ---- Função objetivo ----
    travel_cost = pulp.lpSum(travel[(i, j)] * x[(i, j)][k] for (i, j) in arcs for k in K)
    tardiness_cost = pulp.lpSum(priority_weight[priority[i]] * atraso[i] for i in customer_ids)
    prob += travel_cost + tardiness_weight_scale * tardiness_cost

    # ---- (1)(2) grau de entrada/saída = 1 para cada ponto de atendimento ----
    for j in customer_ids:
        prob += pulp.lpSum(x[(i, j)][k] for i in all_ids if i != j for k in K) == 1, f"in_deg_{j}"
    for i in customer_ids:
        prob += pulp.lpSum(x[(i, j)][k] for j in all_ids if j != i for k in K) == 1, f"out_deg_{i}"

    # ---- (3) conservação de fluxo por equipe em cada ponto de atendimento ----
    for k in K:
        for h in customer_ids:
            inflow = pulp.lpSum(x[(i, h)][k] for i in all_ids if i != h)
            outflow = pulp.lpSum(x[(h, j)][k] for j in all_ids if j != h)
            prob += inflow == outflow, f"flow_{h}_{k}"

    # ---- (4) cada equipe sai/retorna do depósito no máximo uma vez, e balanceado ----
    for k in K:
        out_depot = pulp.lpSum(x[(0, j)][k] for j in customer_ids)
        in_depot = pulp.lpSum(x[(i, 0)][k] for i in customer_ids)
        prob += out_depot <= 1, f"depot_out_{k}"
        prob += in_depot <= 1, f"depot_in_{k}"
        prob += out_depot == in_depot, f"depot_balance_{k}"

    # ---- (5) propagação de tempo / eliminação de sub-rotas (estilo MTZ) ----
    for i in all_ids:
        for j in customer_ids:
            if i == j:
                continue
            start_i = 0 if i == 0 else t[i]
            dur_i = duration[i]
            prob += (
                t[j] >= start_i + dur_i + travel[(i, j)] - BIG_M * (1 - y[(i, j)])
            ), f"time_prop_{i}_{j}"

    # janela de tempo: não pode iniciar antes do earliest_start (equipe pode esperar)
    for i in customer_ids:
        prob += t[i] >= earliest[i], f"earliest_{i}"
        prob += atraso[i] >= t[i] - due[i], f"tardiness_{i}"

    # ---- (7) jornada máxima por equipe ----
    for k in K:
        route_time = pulp.lpSum(travel[(i, j)] * x[(i, j)][k] for (i, j) in arcs)
        service_time = pulp.lpSum(
            duration[i] * pulp.lpSum(x[(i, j)][k] for j in all_ids if j != i)
            for i in customer_ids
        )
        prob += route_time + service_time <= max_minutes[k], f"max_duration_{k}"

    import re
    import tempfile
    import os

    log_fd, log_path = tempfile.mkstemp(suffix=".log")
    os.close(log_fd)
    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_sec, logPath=log_path)
    t0 = time.time()
    prob.solve(solver)
    elapsed = time.time() - t0

    status = pulp.LpStatus[prob.status]

    # Lê o log do CBC para reportar o gap de otimalidade real (independente
    # do rótulo simplificado que o PuLP devolve em LpStatus).
    cbc_log = ""
    try:
        with open(log_path) as f:
            cbc_log = f.read()
    finally:
        try:
            os.remove(log_path)
        except OSError:
            pass

    proven_optimal = "Optimal solution found" in cbc_log and "Stopped on time" not in cbc_log
    lb_match = re.search(r"Lower bound:\s*([\-\d\.]+)", cbc_log)
    gap_match = re.search(r"Gap:\s*([\-\d\.]+)", cbc_log)
    lower_bound = float(lb_match.group(1)) if lb_match else None
    reported_gap = float(gap_match.group(1)) if gap_match else None

    # Extrai rotas
    routes = {k: [] for k in K}
    for k in K:
        used_arcs = [(i, j) for (i, j) in arcs if pulp.value(x[(i, j)][k]) and pulp.value(x[(i, j)][k]) > 0.5]
        if not used_arcs:
            continue
        # reconstrói sequência a partir do depósito
        nxt = {i: j for (i, j) in used_arcs}
        seq = [0]
        cur = 0
        visited = set()
        while cur in nxt and nxt[cur] not in visited:
            cur = nxt[cur]
            if cur == 0:
                break
            seq.append(cur)
            visited.add(cur)
        routes[k] = seq

    result = {
        "status": status,
        "proven_optimal": proven_optimal,
        "lower_bound": lower_bound,
        "optimality_gap_pct": reported_gap,
        "objective": pulp.value(prob.objective),
        "travel_cost": pulp.value(travel_cost),
        "tardiness_cost": pulp.value(tardiness_cost),
        "solve_time_sec": elapsed,
        "routes": routes,
        "arrival_times": {i: pulp.value(t[i]) for i in customer_ids},
        "tardiness": {i: pulp.value(atraso[i]) for i in customer_ids},
        "nodes": nodes,
        "travel_matrix": travel,
        "priority": priority,
        "due": due,
    }
    return result


if __name__ == "__main__":
    from data_generator import build_instance

    points, crews = build_instance(n_points=6, n_crews=2, seed=1)
    res = solve_vrptw(points, crews, time_limit_sec=30, msg=False)
    print("status:", res["status"])
    print("objetivo:", res["objective"])
    print("rotas:", res["routes"])
    print("tempo de solve (s):", round(res["solve_time_sec"], 2))
