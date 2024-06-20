# Practical Assignment


## TODO

- Write data-module
  - Should be configurable (hydra)
  - Should support atomic- and residue-level nodes
  - Should output pt-files that can be used in latter stages

- Implement models
    - GNN with auto-regressive layers
    - GNN with DEQ layers
    - Should be configurable (hydra)
    - Should support gating
    - Should support weight-tied layers
    - Should log metrics (WandB)
    - Should support different normalization schemes
    - Should support dropout
    - First layer should upscale channels

- Implement per-node excitement
    - Excitement is used as a pooling scheme to restrict the number of nodes at
      a given time $t$ that participate in the exchange of information (via
      attention)
    - Inputs: 
        - Node hidden-state
        - Local node-excitement variables
        - Global node-excitement variables
    - Use softmax to pool nodes that participate in exchange (via attention)
      at given time $t$
    - Update all states/variables through exchange (via attention)
