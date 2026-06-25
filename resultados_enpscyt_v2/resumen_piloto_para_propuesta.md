# Resumen piloto para propuesta ENPSCyT 2024

## Base

- Filas: 3214
- Columnas: 208
- Variables recodificadas para análisis relacional: 110

## Variable dependiente

- Apoya aumento: 61.50% ponderado
- No apoya aumento: 27.39% ponderado

## Red de variables

- Umbral de asociación: |rho| >= 0.3
- Nodos: 110
- Aristas: 429
- Componentes: 8
- Método de comunidades: louvain
- Modularidad: 0.745

## Comparación de modelos

| modelo                   |     AIC |   LogLik |   Df_model |    N |   delta_AIC_vs_base |
|:-------------------------|--------:|---------:|-----------:|-----:|--------------------:|
| M1_base_sociodemografico | 3370.45 | -1664.23 |         20 | 2797 |               0     |
| M2_indices_teoricos      | 3208    | -1575    |         28 | 2797 |            -162.452 |
| M3_indices_red           | 3265.24 | -1603.62 |         28 | 2797 |            -105.218 |
| M4_teoricos_y_red        | 3179.32 | -1552.66 |         36 | 2797 |            -191.134 |

## Principales OR - índices teóricos

| variable                              |       OR |   IC95_inf |   IC95_sup |     p_value |
|:--------------------------------------|---------:|-----------:|-----------:|------------:|
| idx_confianza_valoracion_cientificos  | 2.37768  |   1.94987  |    2.89936 | 1.14958e-17 |
| idx_optimismo_tecnocientifico         | 2.1635   |   1.70117  |    2.75148 | 3.14395e-10 |
| idx_contexto_pais                     | 1.06127  |   0.919342 |    1.22511 | 0.41688     |
| idx_institucionalidad_cti             | 1.05779  |   0.887478 |    1.26079 | 0.530486    |
| idx_participacion_cultural_cientifica | 1.00694  |   0.825927 |    1.22761 | 0.945497    |
| idx_percepcion_capacidades_nacionales | 0.868789 |   0.734659 |    1.02741 | 0.100186    |
| idx_actitudes_ia_riesgos_beneficios   | 0.82909  |   0.665285 |    1.03323 | 0.0951362   |
| idx_capital_informativo               | 0.787562 |   0.593659 |    1.0448  | 0.0977085   |

## Principales OR - índices derivados de red

| variable   |       OR |   IC95_inf |   IC95_sup |     p_value |
|:-----------|---------:|-----------:|-----------:|------------:|
| red_com_02 | 1.80232  |   1.55371  |   2.0907   | 7.32449e-15 |
| red_com_01 | 1.66612  |   1.34313  |   2.06678  | 3.43303e-06 |
| red_com_06 | 1.27666  |   1.07644  |   1.51413  | 0.00501212  |
| red_com_07 | 1.01504  |   0.863602 |   1.19305  | 0.856259    |
| red_com_08 | 0.968595 |   0.852295 |   1.10077  | 0.624904    |
| red_com_04 | 0.896992 |   0.77517  |   1.03796  | 0.144372    |
| red_com_05 | 0.847485 |   0.702283 |   1.02271  | 0.0843847   |
| red_com_03 | 0.841311 |   0.711958 |   0.994167 | 0.0424939   |

## Archivos producidos

- piloto_variable_dependiente.csv
- piloto_cobertura_variables.csv
- indices_teoricos_composicion.csv
- matriz_correlaciones_ponderadas.csv
- comunidades_variables.csv
- robustez_red_umbral.csv
- graph_embeddings_espectrales.csv
- modelo_or_indices_teoricos.csv
- modelo_or_indices_red.csv
- comparacion_modelos_v2.csv
- perfiles_ciudadanos_resumen.csv, si sklearn está instalado
- fig_or_indices_teoricos.png
- fig_or_indices_red.png
- fig_red_variables_v2.png
