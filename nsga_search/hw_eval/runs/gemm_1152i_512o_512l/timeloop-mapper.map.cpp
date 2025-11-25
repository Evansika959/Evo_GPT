  // =====================================================================
  // WARNING: this is auto-generated, untested code and will probably need
  // a good amount of massaging to work properly. In specific, please fix 
  // the following:
  // (1) Tiled-tensor shapes will probably not work for sliding windows.
  // (2) Shrink sizes (the 2nd parameter in AddTileLevel()) are incorrect
  //     for sliding windows.
  // (3) Tile-access granularities (3rd/1st parameter in Add/BypassTileLevel()
  //     and multiplier for the 4th/2nd parameter) need to be massaged.
  // (4) Verify that the latency (kXXXLatency) variables are defined.
  // (5) Compute code only contains the tensors. An expression needs to be
  //     filled in.

  Tensor Weights("Weights");
  Tensor Inputs("Inputs");
  Tensor Outputs("Outputs");

  static const int E5 = 2;
  static const int D5 = 4;
  static const int L4 = 2;
  static const int G3 = 1;
  static const int D3s = 16;
  static const int D2 = 8;
  static const int E2 = 36;
  static const int L2 = 16;
  static const int G1 = 1;
  static const int E1s = 16;
  static const int L0 = 16;

  Weights.Resize({ E5, D5, G3, D3s, D2, E2, G1, E1s });
  Inputs.Resize({ E5, L4, G3, E2, L2, G1, E1s, L0 });
  Outputs.Resize({ D5, L4, G3, D3s, D2, L2, G1, L0 });

  Var e5("e5");
  Var d5("d5");
  Var l4("l4");
  Var g3("g3");
  Var d3s("d3s");
  Var d2("d2");
  Var e2("e2");
  Var l2("l2");
  Var g1("g1");
  Var e1s("e1s");
  Var l0("l0");

  // DRAM tiles 
  Weights.AddTileLevel(E5*D5*G3*D3s*D2*E2*G1*E1s*1, E5*D5*G3*D3s*D2*E2*G1*E1s*1, 1, 1 * kBackingStoreLatency);
  Weights.BindCurrentTileLevel("DRAM", 1);
  Inputs.AddTileLevel(E5*L4*G3*E2*L2*G1*E1s*L0*1, E5*L4*G3*E2*L2*G1*E1s*L0*1, 1, 1 * kBackingStoreLatency);
  Inputs.BindCurrentTileLevel("DRAM", 1);
  Outputs.AddTileLevel(D5*L4*G3*D3s*D2*L2*G1*L0*1, D5*L4*G3*D3s*D2*L2*G1*L0*1, 1, 1 * kBackingStoreLatency);
  Outputs.BindCurrentTileLevel("DRAM", 1);

  t_for(e5, 0, E5); {
    t_for(d5, 0, D5); {

      // Scratchpad tiles 
      Weights.AddTileLevel(G3*D3s*D2*E2*G1*E1s*1, G3*D3s*D2*E2*G1*E1s*1, 1, 1 * kDRAMLatency);
      Weights.BindCurrentTileLevel("Scratchpad", 1);
      Inputs.AddTileLevel(L4*G3*E2*L2*G1*E1s*L0*1, L4*G3*E2*L2*G1*E1s*L0*1, 1, 1 * kDRAMLatency);
      Inputs.BindCurrentTileLevel("Scratchpad", 1);
      Outputs.BypassTileLevel(1, 1 * kDRAMLatency);

      t_for(l4, 0, L4); {

        // inter_PERows_spatial tiles 
        Weights.BypassTileLevel(1, 1 * kScratchpadLatency);
        Inputs.BypassTileLevel(1, 1 * kScratchpadLatency);
        Outputs.BypassTileLevel(1, 1 * kScratchpadLatency);

        t_for(g3, 0, G3); {
          s_for(d3s, 0, D3s); {

            // Accumulator tiles 
            Weights.BypassTileLevel(1, 1 * kinter_PERows_spatialLatency);
            Inputs.BypassTileLevel(1, 1 * kinter_PERows_spatialLatency);
            Outputs.AddTileLevel(D2*L2*G1*L0*1, D2*L2*G1*L0*1, 1, 1 * kinter_PERows_spatialLatency);
            Outputs.BindCurrentTileLevel("Accumulator", D3s*1);

            t_for(d2, 0, D2); {
              t_for(e2, 0, E2); {
                t_for(l2, 0, L2); {

                  // inter_PECols_spatial tiles 
                  Weights.BypassTileLevel(1, 1 * kAccumulatorLatency);
                  Inputs.BypassTileLevel(1, 1 * kAccumulatorLatency);
                  Outputs.BypassTileLevel(1, 1 * kAccumulatorLatency);

                  t_for(g1, 0, G1); {
                    s_for(e1s, 0, E1s); {

                      // Registers tiles 
                      Weights.AddTileLevel(1, 1, 1, 1 * kinter_PECols_spatialLatency);
                      Weights.BindCurrentTileLevel("Registers", D3s*E1s*1);
                      Inputs.BypassTileLevel(1, 1 * kinter_PECols_spatialLatency);
                      Outputs.BypassTileLevel(1, 1 * kinter_PECols_spatialLatency);

                      t_for(l0, 0, L0); {

                        // === COMPUTE === fill in a compute expression using the following tensors:
                        Weights[e5][d5][g3][d3s][d2][e2][g1][e1s]; // read-only
                        Inputs[e5][l4][g3][e2][l2][g1][e1s][l0]; // read-only
                        Outputs[d5][l4][g3][d3s][d2][l2][g1][l0]; // read-write

                      } end();
                    } end();
                  } end();
                } end();
              } end();
            } end();
          } end();
        } end();
      } end();
    } end();
  } end();

