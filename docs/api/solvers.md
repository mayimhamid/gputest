# Cases and solvers

## Structured (MAC) solver

::: flowtrace.structured.case.StructuredCase
    options:
      members: false

::: flowtrace.structured.case.Side

::: flowtrace.structured.solver.MACSolver
    options:
      members: [step, residuals, set_sources, fields, centerline, nusselt, max_divergence, state_dict, load_state, describe]

## Unstructured pressure-based solver

::: flowtrace.unstructured.case.UnstructuredCase
    options:
      members: false

::: flowtrace.unstructured.case.BoundarySpec

::: flowtrace.unstructured.simple.SimpleSolver
    options:
      members: [step, residuals, initialize, set_sources, fields, mass_balance, heat_flow, state_dict, load_state, describe]

::: flowtrace.unstructured.conjugate.make_solver

::: flowtrace.unstructured.conjugate.ConjugateSolver
    options:
      members: [step, heat_flow, fields]

## Turbulence models

::: flowtrace.unstructured.turbulence
    options:
      members: [KOmegaSST, KEpsilonRealizable, KEpsilonStandard, spalding_uplus]

## Hardware

::: flowtrace.hardware.backend.get_backend

::: flowtrace.hardware.planner.plan_structured
