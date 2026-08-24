"""
Gerador de dados sintéticos para o problema de roteirização de equipes de campo.

IMPORTANTE: todos os dados aqui são artificiais (gerados por distribuição
aleatória com seed fixa). Nenhum dado real de nenhuma empresa é utilizado.
O cenário é inspirado, em termos de estrutura do problema, no setor de
distribuição de energia elétrica (atendimentos de campo: manutenção,
inspeção, instalação, emergência), mas os números, coordenadas e nomes
são fictícios.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42

SERVICE_TYPES = ["emergencial", "manutencao_preventiva", "instalacao_nova", "inspecao"]
SERVICE_TYPE_WEIGHTS = [0.15, 0.40, 0.25, 0.20]

# Prioridade 1 = mais urgente, 3 = menos urgente
PRIORITY_BY_TYPE = {
    "emergencial": 1,
    "manutencao_preventiva": 2,
    "instalacao_nova": 3,
    "inspecao": 3,
}

# Peso de penalidade por atraso (por minuto), maior para prioridades mais altas
PRIORITY_PENALTY_WEIGHT = {1: 8.0, 2: 3.0, 3: 1.0}

WORKDAY_START_MIN = 0        # 08:00 -> minuto 0 (referência)
WORKDAY_END_MIN = 540        # 17:00 -> 9h de jornada (540 min)


def generate_service_points(n_points: int, area_km: float = 22.0, seed: int = RNG_SEED) -> pd.DataFrame:
    """Gera N pontos de atendimento distribuídos numa área quadrada de `area_km` x `area_km`."""
    rng = np.random.default_rng(seed)

    ids = [f"P{i+1:02d}" for i in range(n_points)]
    x = rng.uniform(0, area_km, n_points)
    y = rng.uniform(0, area_km, n_points)

    service_type = rng.choice(SERVICE_TYPES, size=n_points, p=SERVICE_TYPE_WEIGHTS)
    priority = np.array([PRIORITY_BY_TYPE[s] for s in service_type])

    # Duração do atendimento (minutos), varia por tipo de serviço
    duration_base = {
        "emergencial": (30, 60),
        "manutencao_preventiva": (40, 90),
        "instalacao_nova": (60, 120),
        "inspecao": (20, 45),
    }
    duration_min = np.array([
        rng.integers(duration_base[s][0], duration_base[s][1] + 1) for s in service_type
    ])

    # Janela de atendimento (soft): início possível dentro do dia útil,
    # prazo desejado (due) até o qual o atendimento deveria começar.
    earliest_start = rng.integers(WORKDAY_START_MIN, WORKDAY_END_MIN - 120, n_points)
    # Urgentes têm prazo mais apertado
    slack_by_priority = {1: (30, 90), 2: (90, 180), 3: (150, 300)}
    due = np.array([
        earliest_start[i] + rng.integers(*slack_by_priority[priority[i]])
        for i in range(n_points)
    ])
    due = np.minimum(due, WORKDAY_END_MIN)

    df = pd.DataFrame({
        "id": ids,
        "x_km": x.round(2),
        "y_km": y.round(2),
        "service_type": service_type,
        "priority": priority,
        "duration_min": duration_min,
        "earliest_start_min": earliest_start,
        "due_min": due,
    })
    return df


def generate_crews(n_crews: int, depot_xy=(11.0, 11.0), max_hours: float = 9.0) -> pd.DataFrame:
    """Gera equipes de campo, todas partindo de uma base central (depósito único)."""
    ids = [f"E{i+1}" for i in range(n_crews)]
    df = pd.DataFrame({
        "crew_id": ids,
        "depot_x_km": depot_xy[0],
        "depot_y_km": depot_xy[1],
        "max_minutes": max_hours * 60,
    })
    return df


def build_instance(n_points: int, n_crews: int, seed: int = RNG_SEED):
    points = generate_service_points(n_points, seed=seed)
    crews = generate_crews(n_crews)
    return points, crews


if __name__ == "__main__":
    points, crews = build_instance(n_points=15, n_crews=3)
    points.to_csv("data/service_points.csv", index=False)
    crews.to_csv("data/crews.csv", index=False)
    print(points)
    print(crews)
