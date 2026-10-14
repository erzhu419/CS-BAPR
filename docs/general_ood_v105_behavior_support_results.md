# v105: additional standard references improve mean predictions

Task t141069 completed in 66.5 seconds: 62 windows, 3548 transitions and 186
cached-incumbent RMS comparisons passed. Only 612 KB of scalar JSON was synced.

On six old coverage failures, normalized prediction RMS falls from 0.29524 to
0.23700 with standard collection (-19.7%, all six improve). Exploratory collection
gives 0.25649 (-13.1%), 8.2% worse than the budget-matched standard arm.
New standard references supply 21.1% of the 64-neighbor selections on these
windows; exploratory references supply 25.0%, but greater neighbor share alone
does not produce a better mean.

Standard collection reduces changed updating RMS by 13.9% (23/24 windows) and
frozen RMS by 3.8% (24/24). All three updating seeds improve: 6.0%, 11.7%, 20.0%.
Historical slippery-tail RMS decreases 21.8%. Pooled normal RMS improves 1.4%,
but six of twelve normal windows regress locally.

Advance only the standard mean to matched normal calibration and complete
belief replay with the full incumbent covariance and old moment center frozen.
Retain v98 until recognition and normal-risk controls pass. No actor integration.

## Limitations

Standard/exploratory budgets match each other, not the smaller incumbent.
Improvement versus v98 combines extra reference rows and new collection seeds;
it does not establish a benefit of action perturbation. These are inspected
development windows; prediction/support gains are not recognition or RL-return gains.
