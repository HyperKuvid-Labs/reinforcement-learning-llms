# vibecoded

from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Console
from rich import box
from rich.rule import Rule
from rich.columns import Columns
from rich.align import Align
import time

from train import (
    advantage_fns, train_experiment,
    num_epochs, num_classes, k, learning_rate,
    device, use_amp, train_dataset, val_dataset,
)

console = Console()

# ── shared state ──────────────────────────────────────────
state = {
    "adv_name":       "",
    "adv_idx":        0,
    "total_advs":     len(advantage_fns),
    "epoch":          0,
    "total_epochs":   num_epochs,
    "batch":          0,
    "total_batches":  1,
    "loss":           0.0,
    "lr":             learning_rate,
    "step_losses":    [],        # rolling last-N step losses for sparkline
    "epoch_losses":   [],        # avg loss per completed epoch (current adv)
    "results":        [],        # {adv, epoch, avg_loss, val_acc}
    "done":           [],        # finished adv names
}

ADV_COLORS = {"grpo": "bright_cyan", "reinforce": "bright_magenta", "maxrl": "bright_yellow"}
ADV_ICONS  = {"grpo": "◈", "reinforce": "◉", "maxrl": "◆"}


# ── helpers ───────────────────────────────────────────────
def sparkline(values: list, width: int = 36) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    vals = values[-width:] if values else []
    if not vals:
        return "[dim]" + ("─" * width) + "[/dim]"
    mn, mx = min(vals), max(vals)
    if mn == mx:
        chars = blocks[0] * len(vals)
    else:
        chars = "".join(blocks[int((v - mn) / (mx - mn) * (len(blocks) - 1))] for v in vals)
    return f"[cyan]{chars}[/cyan]"


def bar(current: int, total: int, width: int = 38, color: str = "cyan") -> str:
    if total == 0:
        return f"[dim]{'─' * width}[/dim]  0/0"
    filled = int(width * current / total)
    b = f"[{color}]{'█' * filled}[/{color}][dim]{'░' * (width - filled)}[/dim]"
    pct = 100 * current / total
    return f"{b}  [bold]{current}/{total}[/bold] [dim]({pct:.0f}%)[/dim]"


# ── renderers ─────────────────────────────────────────────
def make_header() -> Panel:
    t = Text(justify="center")
    t.append("⚡  RL Training  ", style="bold bright_cyan")
    t.append("CIFAR-100", style="bold white")
    t.append("  ·  ResNet-18  ·  AMP  ·  GRPO / REINFORCE / MaxRL", style="dim white")
    return Panel(t, box=box.DOUBLE_EDGE, style="bright_cyan", padding=(0, 2))


def make_config() -> Panel:
    g = Table.grid(padding=(0, 2))
    g.add_column(style="dim")
    g.add_column(style="bold white")

    s = state
    adv = s["adv_name"].upper() if s["adv_name"] else "—"
    color = ADV_COLORS.get(s["adv_name"], "white")
    icon  = ADV_ICONS.get(s["adv_name"], "·")

    g.add_row("dataset",   "CIFAR-100")
    g.add_row("model",     "ResNet-18")
    g.add_row("classes",   str(num_classes))
    g.add_row("rollouts",  str(k))
    g.add_row("amp",       "[green]enabled[/green]" if use_amp else "[red]disabled[/red]")
    g.add_row("device",    str(device))
    g.add_row("lr",        f"{s['lr']:.2e}")
    g.add_row("",          "")
    g.add_row("training",  f"[{color}]{icon}  {adv}[/{color}]  [dim]{s['adv_idx']}/{s['total_advs']}[/dim]")
    g.add_row("epoch",     f"[bold]{s['epoch'] + 1}[/bold] / {s['total_epochs']}")

    return Panel(g, title="[bold dim]config[/bold dim]", box=box.ROUNDED, padding=(1, 2))


def make_results_table() -> Panel:
    t = Table(
        box=box.SIMPLE_HEAD,
        header_style="bold dim",
        show_lines=False,
        expand=True,
        padding=(0, 1),
    )
    t.add_column("method",   style="bold", width=12)
    t.add_column("epoch",    justify="right", width=8)
    t.add_column("avg loss", justify="right", width=10)
    t.add_column("val acc",  justify="right", width=10)

    for r in state["results"]:
        color = ADV_COLORS.get(r["adv"], "white")
        icon  = ADV_ICONS.get(r["adv"], "·")
        adv_rows = [x["val_acc"] for x in state["results"] if x["adv"] == r["adv"]]
        acc_style = "bold green" if adv_rows and r["val_acc"] == max(adv_rows) else ""
        t.add_row(
            f"[{color}]{icon} {r['adv'].upper()}[/{color}]",
            f"{r['epoch'] + 1}/{num_epochs}",
            f"{r['avg_loss']:.4f}",
            f"[{acc_style}]{r['val_acc']:.4f}[/{acc_style}]",
        )

    if not state["results"]:
        t.add_row("[dim]—[/dim]", "[dim]—[/dim]", "[dim]—[/dim]", "[dim]—[/dim]")

    return Panel(t, title="[bold dim]epoch results[/bold dim]", box=box.ROUNDED, padding=(0, 1))


def make_progress() -> Panel:
    s = state
    color = ADV_COLORS.get(s["adv_name"], "cyan")
    icon  = ADV_ICONS.get(s["adv_name"], "·")
    adv   = s["adv_name"].upper() if s["adv_name"] else "—"

    epoch_bar = bar(s["epoch"] + 1,  s["total_epochs"],   width=36, color=color)
    batch_bar = bar(s["batch"],       s["total_batches"],  width=36, color="white")

    spark = sparkline(s["step_losses"], width=36)

    g = Table.grid(padding=(0, 1))
    g.add_column(style="dim", width=8)
    g.add_column()

    g.add_row("epoch",     epoch_bar)
    g.add_row("batch",     batch_bar)
    g.add_row("loss now",  f"[bold {color}]{s['loss']:.4f}[/bold {color}]")
    g.add_row("trend",     spark)

    done_text = "  ".join(
        f"[{ADV_COLORS.get(d, 'white')}]{ADV_ICONS.get(d, '·')} {d.upper()} ✓[/{ADV_COLORS.get(d, 'white')}]"
        for d in s["done"]
    ) or "[dim]none yet[/dim]"
    g.add_row("done",      done_text)

    title = f"[bold {color}]{icon}  {adv}[/bold {color}]  [dim]·  epoch {s['epoch'] + 1}/{s['total_epochs']}[/dim]"
    return Panel(g, title=title, box=box.ROUNDED, padding=(1, 2))


def build_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header",   size=3),
        Layout(name="body"),
        Layout(name="progress", size=9),
    )
    layout["body"].split_row(
        Layout(name="config",  ratio=1),
        Layout(name="results", ratio=2),
    )
    return layout


def render(layout: Layout) -> None:
    layout["header"].update(make_header())
    layout["config"].update(make_config())
    layout["results"].update(make_results_table())
    layout["progress"].update(make_progress())


# ── callbacks ─────────────────────────────────────────────
def on_batch(batch_idx, total_batches, loss):
    state["batch"]         = batch_idx
    state["total_batches"] = total_batches
    state["loss"]          = loss
    state["step_losses"].append(loss)
    if len(state["step_losses"]) > 80:
        state["step_losses"].pop(0)


def on_epoch(epoch, total_epochs, avg_loss, val_acc, lr):
    state["epoch"]        = epoch
    state["total_epochs"] = total_epochs
    state["lr"]           = lr
    state["epoch_losses"].append(avg_loss)
    state["results"].append({
        "adv":      state["adv_name"],
        "epoch":    epoch,
        "avg_loss": avg_loss,
        "val_acc":  val_acc,
    })


def on_done(adv_name):
    state["done"].append(adv_name)


callbacks = {
    "on_batch": on_batch,
    "on_epoch": on_epoch,
    "on_done":  on_done,
}


# ── main ──────────────────────────────────────────────────
def main():
    layout = build_layout()

    with Live(layout, console=console, refresh_per_second=8, screen=True):
        for idx, (adv_name, advantage_fn) in enumerate(advantage_fns.items()):
            state["adv_name"]    = adv_name
            state["adv_idx"]     = idx + 1
            state["epoch"]       = 0
            state["batch"]       = 0
            state["step_losses"] = []
            state["epoch_losses"] = []
            state["loss"]        = 0.0
            state["lr"]          = learning_rate

            render(layout)
            train_experiment(adv_name, advantage_fn, callbacks=callbacks)

    # ── final summary ──────────────────────────────────────
    console.print()
    console.print(Rule("[bold bright_cyan]Training Complete[/bold bright_cyan]"))
    console.print()

    summary = Table(
        title="[bold]Final Results[/bold]",
        box=box.ROUNDED,
        header_style="bold dim",
        show_lines=True,
        padding=(0, 2),
    )
    summary.add_column("method",    style="bold", width=14)
    summary.add_column("best loss", justify="right")
    summary.add_column("best acc",  justify="right")
    summary.add_column("final acc", justify="right")

    for adv in advantage_fns:
        rows = [r for r in state["results"] if r["adv"] == adv]
        if not rows:
            continue
        color = ADV_COLORS.get(adv, "white")
        icon  = ADV_ICONS.get(adv, "·")
        best_loss = min(r["avg_loss"] for r in rows)
        best_acc  = max(r["val_acc"]  for r in rows)
        final_acc = rows[-1]["val_acc"]
        summary.add_row(
            f"[{color}]{icon} {adv.upper()}[/{color}]",
            f"{best_loss:.4f}",
            f"[bold green]{best_acc:.4f}[/bold green]",
            f"{final_acc:.4f}",
        )

    console.print(Align.center(summary))
    console.print()


if __name__ == "__main__":
    main()
