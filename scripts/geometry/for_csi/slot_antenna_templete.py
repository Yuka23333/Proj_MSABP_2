"""Top-down view of a slot antenna: a square substrate with a thin slot cut in the middle."""

import matplotlib.pyplot as plt
from shapely.geometry import box

SUBSTRATE_SIZE = 50

SLOT_LENGTH = 40
SLOT_WIDTH = 2

# In a standard slot antenna, "outer" (the topological outer boundary) and "substrate" (the
# physical edge) coincide -- nudged apart here so both are visible as separate lines.
OUTER_SUBSTRATE_GAP = 0.2
BOUNDARY_LINEWIDTH = 2

substrate = box(-SUBSTRATE_SIZE / 2, -SUBSTRATE_SIZE / 2, SUBSTRATE_SIZE / 2, SUBSTRATE_SIZE / 2)
outer_boundary = box(
    -SUBSTRATE_SIZE / 2 - OUTER_SUBSTRATE_GAP, -SUBSTRATE_SIZE / 2 - OUTER_SUBSTRATE_GAP,
    SUBSTRATE_SIZE / 2 + OUTER_SUBSTRATE_GAP, SUBSTRATE_SIZE / 2 + OUTER_SUBSTRATE_GAP,
)
slot = box(-SLOT_LENGTH / 2, -SLOT_WIDTH / 2, SLOT_LENGTH / 2, SLOT_WIDTH / 2)

print("Substrate:", substrate)
print("Substrate Bounds:", substrate.bounds)
print("Outer boundary:", outer_boundary)
print("Outer boundary Bounds:", outer_boundary.bounds)
print("Slot:", slot)
print("Slot Bounds:", slot.bounds)

fig, ax = plt.subplots()

x, y = substrate.exterior.xy
ax.fill(x, y, alpha=0.3, color="tab:green", label="metal")
ax.plot(x, y, color="black", linewidth=BOUNDARY_LINEWIDTH, label="substrate")

x, y = outer_boundary.exterior.xy
ax.plot(x, y, color="dimgray", linewidth=BOUNDARY_LINEWIDTH, label="outer")

x, y = slot.exterior.xy
ax.fill(x, y, alpha=0.8, color="white")
ax.plot(x, y, color="tab:red", linewidth=BOUNDARY_LINEWIDTH, label="inner")

ax.legend(loc="upper right")
ax.set_aspect("equal")
ax.set_title("Slot antenna template (top view)")

plt.show()
