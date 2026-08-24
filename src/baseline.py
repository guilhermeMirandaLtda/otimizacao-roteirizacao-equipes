"""
Baseline "despachante humano": heurística simples e ingênua, para representar
como a alocação costuma ser feita manualmente (sem otimização) — round-robin
por prioridade/prazo entre as equipes, e depois vizinho-mais-próximo dentro
de cada equipe. Serve de referência para medir o ganho real do modelo MILP.
"""

import math

import pandas as pd

from model import build_nodes, build_travel_time_matrix, AVG_SPEED_KMH


def nearest_neighbor_route(depot_id, customer_ids, nodes):
    route = [depot_id]
    remaining = set(customer_ids)
    current = depot_id
    while remaining:
        nxt = min(remaining, key=lambda c: math.hypot(
            nodes[current][0] - nodes[c][0], nodes[current][1] - nodes[c][1]))
        route.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return route


def solve_baseline(points: pd.DataFrame, crews: pd.DataFrame):
    depot_xy = (crews.iloc[0]["depot_x_km"], crews.iloc[0]["depot_y_km"])
    nodes = build_nodes(points, depot_xy)
    travel = build_travel_time_matrix(nodes)

    # Ordena pontos por prazo (due) - prioridade de despacho "óbvia" que um
    # despachante manual aplicaria, sem otimizar rotas nem carga entre equipes
    points_sorted = points.sort_values("due_min")
    crew_ids = list(crews["crew_id"])
    n_crews = len(crew_ids)

    # Round-robin simples entre equipes (sem considerar geografia)
    assignment = {k: [] for k in crew_ids}
    for idx, (_, row) in enumerate(points_sorted.iterrows()):
        crew = crew_ids[idx % n_crews]
        assignment[crew].append(int(row["id"][1:]))

    duration = {int(r["id"][1:]): r["duration_min"] for _, r in points.iterrows()}
    due = {int(r["id"][1:]): r["due_min"] for _, r in points.iterrows()}
    priority = {int(r["id"][1:]): r["priority"] for _, r in points.iterrows()}
    priority_weight = {1: 8.0, 2: 3.0, 3: 1.0}

    routes = {}
    total_travel = 0.0
    arrival_times = {}
    tardiness = {}
    tardiness_cost = 0.0

    for k in crew_ids:
        custs = assignment[k]
        if not custs:
            routes[k] = []
            continue
        route = nearest_neighbor_route(0, custs, nodes)
        routes[k] = route

        t_cursor = 0.0
        for a, b in zip(route[:-1], route[1:]):
            t_cursor += travel[(a, b)]
            total_travel += travel[(a, b)]
            if b != 0:
                t_cursor = max(t_cursor, 0)  # sem espera modelada explicitamente
                arrival_times[b] = t_cursor
                t_cursor += duration[b]
                late = max(0.0, arrival_times[b] - due[b])
                tardiness[b] = late
                tardiness_cost += priority_weight[priority[b]] * late

    objective = total_travel + tardiness_cost
    return {
        "status": "Heuristica (round-robin + vizinho mais proximo)",
        "objective": objective,
        "travel_cost": total_travel,
        "tardiness_cost": tardiness_cost,
        "routes": routes,
        "arrival_times": arrival_times,
        "tardiness": tardiness,
        "nodes": nodes,
        "travel_matrix": travel,
        "priority": priority,
        "due": due,
    }


if __name__ == "__main__":
    from data_generator import build_instance

    points, crews = build_instance(n_points=15, n_crews=3, seed=42)
    res = solve_baseline(points, crews)
    print("objetivo (baseline):", res["objective"])
    print("travel_cost:", res["travel_cost"])
    print("tardiness_cost:", res["tardiness_cost"])
    print("rotas:", res["routes"])
