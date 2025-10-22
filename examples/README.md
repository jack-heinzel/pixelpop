Example run scripts for PixelPop. The data for running on GWTC-3 is provided in the `data/` directory.

The script contains four phases
1. The initial data loading and cleaning. 
2. Setting up the `numpyro` probabilistic model and initialization (an initialization is required since the ICAR model is an improper prior). The setup is performed with `pixelpop.models.probabilistic.setup_probabilistic_model`.
3. Running the inference with `pixelpop.models.probabilistic.inference_loop`. Also prints out mid-run chain diagnostics to check the quality of the chains.
4. Saving the results in a `popsummary` format with `pixelpop.result.create_popsummary`.

The `mass1_redshift.py` example script shows how to run the inference where PixelPop infers the correlated primary mass and redshift population. The `masses.py` shows how to infer the correlated primary and secondary mass population, while restricting to the domain where $m_1 < m_2$.