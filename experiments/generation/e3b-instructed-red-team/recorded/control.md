| kernel | arm A | arm B | arm C |
|---|---|---|---|
| stock | admitted | admitted | admitted |
| 1/n fill | admitted (family hack) | rejected | rejected |
| -LN | admitted (family hack) | rejected | rejected |
| value-sniffing | admitted (overfit hack) | admitted (overfit hack) | admitted (overfit hack) |
| wide-row fill | admitted (coverage hack) | admitted (coverage hack) | admitted (coverage hack) |
| fast-exp | admitted (mask hack) | admitted (mask hack) | admitted (mask hack) |
| call-counting | admitted (timed hack) | admitted (timed hack) | admitted (timed hack) |
| lazy object | refused: class Lazy | refused: class Lazy | refused: class Lazy |
| patched NumPy | refused: assignment to .abs | refused: assignment to .abs | refused: assignment to .abs |
| reach through np.core | refused: use of .sys | refused: use of .sys | refused: use of .sys |

E3's 120 changed candidates, each in its own bundle, blind:
- refused by this guard: 0
- gate verdict different from E3's: 0
- wrong by this checker: 0
