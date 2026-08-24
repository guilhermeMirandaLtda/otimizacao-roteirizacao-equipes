"""Visualizações: mapas de rotas (MILP vs baseline) e comparação de custos."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CREW_COLORS = ["#1F77B4", "#D62728", "#2CA02C", "#9467BD", "#FF7F0E"]

PRIORITY_LABEL = {1: "Urgente", 2: "Normal", 3: "Baixa"}


def plot_routes(result, title, ax=None, save_path=None):
    nodes = result["nodes"]
    routes = result["routes"]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
        created_fig = True

    depot = nodes[0]
    ax.scatter([depot[0]], [depot[1]], marker="s", s=180, color="black", zorder=5, label="Base (depósito)")

    priority = result.get("priority", {})
    for nid, (x, y) in nodes.items():
        if nid == 0:
            continue
        p = priority.get(nid, 2)
        marker_size = 90 if p == 1 else 55
        ax.scatter([x], [y], s=marker_size, color="#555555", zorder=4,
                   edgecolors="white", linewidths=0.6)
        ax.annotate(f"P{nid}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)

    for idx, (crew, route) in enumerate(routes.items()):
        if not route or len(route) < 2:
            continue
        color = CREW_COLORS[idx % len(CREW_COLORS)]
        xs = [nodes[n][0] for n in route] + [nodes[0][0]]
        ys = [nodes[n][1] for n in route] + [nodes[0][1]]
        ax.plot(xs, ys, "-", color=color, linewidth=2, alpha=0.85, label=f"Equipe {crew}", zorder=3)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("km (leste-oeste)")
    ax.set_ylabel("km (norte-sul)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_aspect("equal", adjustable="box")

    if created_fig and save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)


def plot_comparison(milp_result, baseline_result, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))

    plot_routes(baseline_result, "Baseline: despacho manual\n(round-robin + vizinho mais próximo)", ax=axes[0])
    plot_routes(milp_result, "Modelo MILP (otimizado)", ax=axes[1])

    ax = axes[2]
    categories = ["Custo de\ndeslocamento", "Custo de\natraso (SLA)", "Custo\ntotal"]
    baseline_vals = [baseline_result["travel_cost"], baseline_result["tardiness_cost"], baseline_result["objective"]]
    milp_vals = [milp_result["travel_cost"], milp_result["tardiness_cost"], milp_result["objective"]]

    x = range(len(categories))
    width = 0.35
    bars1 = ax.bar([i - width / 2 for i in x], baseline_vals, width, label="Baseline (manual)", color="#B0B0B0")
    bars2 = ax.bar([i + width / 2 for i in x], milp_vals, width, label="MILP (otimizado)", color="#1F77B4")

    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.1f}", (b.get_x() + b.get_width() / 2, h), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=8)

    reduction = (1 - milp_result["objective"] / baseline_result["objective"]) * 100
    ax.set_title(f"Comparação de custo\n(redução de {reduction:.1f}% no custo total)", fontsize=12, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories)
    ax.set_ylabel("Custo (unidades ponderadas: minutos + penalidade de atraso)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return reduction


if __name__ == "__main__":
    import pickle
    from data_generator import build_instance
    from model import solve_vrptw
    from baseline import solve_baseline

    points, crews = build_instance(n_points=15, n_crews=3, seed=42)
    milp = solve_vrptw(points, crews, time_limit_sec=60, msg=False)
    base = solve_baseline(points, crews)
    reduction = plot_comparison(milp, base, save_path="../outputs/comparacao_rotas.png")
    print("Redução de custo total:", round(reduction, 1), "%")
