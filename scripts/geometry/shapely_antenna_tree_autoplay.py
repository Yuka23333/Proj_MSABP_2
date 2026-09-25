"""Autoplay stress test for shapely_antenna_tree_demo.py.

Drives the real GUI with random edits (add a branch / delete a subtree / slide a K)
through its own handlers, and after every step checks that
  - the Treeview shows exactly the branches in the model,
  - no branch leaves the metal patch or cuts into the feed region (CPW slot + pin),
  - no branch crosses the Y axis (the tree lives in x >= 0 before mirroring).

    python shapely_antenna_tree_autoplay.py                   # watchable: 0.5 s per step
    python shapely_antenna_tree_autoplay.py --pause 1.5       # slower, easier on the eyes
    python shapely_antenna_tree_autoplay.py --pause 0         # as fast as it can draw
    python shapely_antenna_tree_autoplay.py --pause 0 --exit  # unattended check, exit code 0/1

Space pauses / resumes. The window stays open at the end (or on the first failure)
unless --exit is given.
"""

import argparse
import random
import sys
import tkinter as tk

import shapely_antenna_tree_demo as demo

TOLERANCE = 1e-9

# Edit mix. Adding outweighs deleting so the tree keeps growing instead of being
# pruned back to the root; a delete removes a whole subtree.
ADD_PROBABILITY = 0.5
DELETE_PROBABILITY = 0.1  # the remaining 0.4 slides one K of a random branch


def treeview_ids(tree, item=""):
    ids = []
    for child in tree.get_children(item):
        ids += [child, *treeview_ids(tree, child)]
    return ids


def check(app):
    """Return every invariant violation of the current state (empty list = all good)."""
    problems = []
    shown = set(treeview_ids(app.tree)) - {demo.ROOT}
    if shown != set(app.tree_nodes):
        problems.append(f"Treeview/model mismatch: {sorted(shown ^ set(app.tree_nodes))}")
    shapes = demo.build(app.params, app.tree_nodes)
    for node_id, info in shapes["branches"].items():
        outside = info["branch"].difference(shapes["Patch"]).area
        if outside > TOLERANCE:
            problems.append(f"{node_id} leaves the patch by {outside:.3g} mm^2")
        in_feed = info["branch"].intersection(shapes["Feed_Region"]).area
        if in_feed > TOLERANCE:
            problems.append(f"{node_id} cuts into the feed region by {in_feed:.3g} mm^2")
        min_x = info["branch"].bounds[0]
        if min_x < -TOLERANCE:
            problems.append(f"{node_id} crosses the Y axis (min x = {min_x:.3g})")
    return problems


def random_step(app, rng):
    """Apply one random edit the way a user would; return (kind, description)."""
    node = rng.choice([demo.ROOT, *app.tree_nodes])
    app.tree.selection_set(node)
    app.root.update()  # deliver <<TreeviewSelect>> exactly like a mouse click

    roll = rng.random()
    if roll < ADD_PROBABILITY or node == demo.ROOT:
        app.add_child(rng.randrange(2))
        return "add", f"add {app.selected}"
    if roll < ADD_PROBABILITY + DELETE_PROBABILITY:
        doomed = sum(1 for k in app.tree_nodes if k == node or k.startswith(f"{node}/"))
        app.delete_selected()
        return "delete", f"delete {node} (-{doomed})"
    i = rng.randrange(3)
    low, high = demo.k_range(app.tree_nodes, node, i)
    value = rng.uniform(low, high)
    app.k_vars[i].set(value)
    app.on_k_slide(i)
    return "slide", f"{node} K{i + 1} = {value:.2f}"


class Autoplay:
    def __init__(self, app, steps, pause, seed, exit_when_done):
        self.app = app
        self.steps = steps
        self.pause_ms = int(round(pause * 1000))
        self.exit_when_done = exit_when_done
        self.rng = random.Random(seed)
        self.step = 0
        self.paused = False
        self.pending = None
        self.failed = False
        self.counts = {"add": 0, "delete": 0, "slide": 0}
        self.peak_branches = len(app.tree_nodes)
        self.peak_depth = self.depth()

        app.root.bind("<space>", self.toggle_pause)
        problems = check(app)
        if problems:
            self.fail("initial state", problems)
        else:
            self.pending = app.root.after(500, self.tick)  # let the window appear first

    def depth(self):
        return max((k.count("/") + 1 for k in self.app.tree_nodes), default=0)

    def set_title(self, text):
        self.app.root.title(f"Tree autoplay | {text}")

    def toggle_pause(self, _event=None):
        if self.failed or self.step >= self.steps:
            return "break"
        self.paused = not self.paused
        if self.paused:
            if self.pending is not None:
                self.app.root.after_cancel(self.pending)
                self.pending = None
            self.set_title(f"PAUSED at step {self.step}/{self.steps} (space to resume)")
        else:
            self.pending = self.app.root.after(0, self.tick)
        return "break"

    def tick(self):
        self.pending = None
        self.step += 1
        kind, action = random_step(self.app, self.rng)
        self.counts[kind] += 1
        self.peak_branches = max(self.peak_branches, len(self.app.tree_nodes))
        self.peak_depth = max(self.peak_depth, self.depth())
        self.app.root.update_idletasks()  # flush the plot's draw_idle even at --pause 0

        status = f"step {self.step}/{self.steps} | {len(self.app.tree_nodes)} branches | {action}"
        print(status)
        self.set_title(status)

        problems = check(self.app)
        if problems:
            self.fail(f"step {self.step} ({action})", problems)
        elif self.step >= self.steps:
            self.finish()
        else:
            self.pending = self.app.root.after(self.pause_ms, self.tick)

    def fail(self, where, problems):
        self.failed = True
        print(f"FAIL at {where}:")
        for problem in problems:
            print(f"  - {problem}")
        self.set_title(f"FAIL at {where}: {problems[0]}")
        if self.exit_when_done:
            self.app.root.after(0, self.app.root.destroy)

    def finish(self):
        summary = (
            f"PASS: {self.step} steps ({self.counts['add']} adds, {self.counts['delete']} subtree "
            f"deletes, {self.counts['slide']} K slides), all invariants held at every step. "
            f"Peak {self.peak_branches} branches, deepest {self.peak_depth} levels; "
            f"{len(self.app.tree_nodes)} branches at the end."
        )
        print(summary)
        self.set_title(
            f"PASS | {self.step} steps | peak {self.peak_branches} branches, depth {self.peak_depth}"
        )
        if self.exit_when_done:
            self.app.root.after(0, self.app.root.destroy)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pause", type=float, default=0.5,
                        help="seconds between steps; raise it to follow along, 0 = full speed (default 0.5)")
    parser.add_argument("--steps", type=int, default=150, help="number of random edits (default 150)")
    parser.add_argument("--seed", type=int, default=11, help="random seed; same seed = same run (default 11)")
    parser.add_argument("--exit", action="store_true", help="close the window when done; exit code 1 on failure")
    args = parser.parse_args(argv)
    if args.pause < 0:
        parser.error("--pause must be >= 0")
    if args.steps < 1:
        parser.error("--steps must be >= 1")

    root = tk.Tk()
    app = demo.AntennaDemo(root)
    autoplay = Autoplay(app, args.steps, args.pause, args.seed, args.exit)
    root.mainloop()
    return 1 if autoplay.failed else 0


if __name__ == "__main__":
    sys.exit(main())
