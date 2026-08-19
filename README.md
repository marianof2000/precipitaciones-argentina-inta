# Precipitaciones Argentina

Proyecto Python para integrar observaciones de precipitación de estaciones argentinas y,
por etapas, construir un mapa geoespacial y temporal publicable como sitio estático.

## Arquitectura y dependencias

El paquete `src/precipitaciones_argentina` separa catálogo, loaders, validación, normalización,
tiempo, estadísticas, geometría, cobertura, auditoría, visualización y CLI. `pandas`, `xlrd` y
`openpyxl` leen Excel; `pyarrow` genera Parquet; NumPy/SciPy implementan IDW; GeoPandas/Shapely
procesan geometrías; Pillow codifica los rásteres; Folium/Leaflet construyen el mapa. `pytest` y
Ruff son dependencias de desarrollo. Todas se resuelven exclusivamente mediante `uv.lock`.

## Requisitos e instalación

Requiere `uv` y Python 3.14.6. `uv` gestiona Python, el entorno y todas las dependencias:

```bash
uv sync --frozen
```

No se usa `requirements.txt`.

## Datos y catálogo

Los Excel originales permanecen sin modificaciones en `datos/`. Sólo se procesan los archivos
declarados en `datos/estaciones.json`; el programa nunca descubre XLS indiscriminadamente.

El catálogo actual conserva los metadatos originales de INTA. Su sección `configuracion` define
la fuente, hoja, unidad y mapeo explícito de columnas compartido. Cada elemento de `estaciones`
declara `id_estacion`, `metadata_origen` (nombre, localidad, provincia, latitud y longitud) y
`descarga.archivo`. También se admite el formato extensible `datasets`, donde cada entrada puede
declarar `id`, `archivo`, `fuente`, `estacion`, `localidad`, `provincia`, coordenadas, `hoja`,
`unidad_precipitacion`, `campos` y metadatos adicionales.

Para incorporar una estación estándar:

1. Copiar el `.xls` o `.xlsx` sin renombrarlo a `datos/`.
2. Declarar exactamente ese archivo en el catálogo.
3. Declarar hoja, unidad, columnas y metadatos; no es necesario modificar Python.

Los XLS actuales usan la hoja `Datos diarios`, la columna `Fecha` y
`Precipitacion_Pluviometrica`; estos nombres no se infieren, sino que están declarados en el
catálogo. Un dataset con otra estructura debe declarar sus propios `campos` y `hoja`.

## Procesamiento

La Etapa 1 lee y valida el catálogo, comprueba archivo/hoja/columnas, normaliza fechas,
coordenadas y unidades, descarta explícitamente registros inválidos o duplicados, y agrega por
estación y trimestre. El modelo conserva dataset, archivo, fuente, estación, ubicación, fecha,
año, trimestre, período, valor/unidad original y milímetros.

Los trimestres son T1 (enero-marzo), T2 (abril-junio), T3 (julio-septiembre) y T4
(octubre-diciembre). La variable científica publicada es precipitación acumulada y, por ello,
la agregación trimestral se restringe explícitamente a `sum`;
un dato ausente nunca se reemplaza por cero.

```bash
uv run precipitaciones
```

La Etapa 1 genera `output/datos_normalizados.parquet`, con los datos trimestrales utilizados por
las siguientes etapas. La Etapa 2 genera `output/index.html`: incluye límites provinciales,
observaciones reales, cobertura de estaciones, selector de los 303 períodos, estadísticas y una
escala cromática global desde 0 hasta el máximo real, con cortes conceptuales cada 10 mm.

La Etapa 3 incorpora superficies IDW trimestrales. La grilla y su resolución se configuran en
`config.py`. La resolución productiva es `0.1°`; `0.05°` ofrece más detalle, pero aumenta
considerablemente tiempo, memoria y tamaño del HTML. `SPATIAL_DEBUG` puede generar máscaras,
alpha, RGBA e IDW auxiliares para un período elegido. Cada superficie se restringe al territorio
y a una distancia máxima provisional de
350 km respecto de una estación activa del período. El convex hull no actúa como frontera. Una
estación activa tiene coordenadas, precipitación válida y ubicación dentro de la máscara
territorial; su provincia declarada no la excluye. Con menos de tres ubicaciones la estimación
queda transparente como “Sin datos suficientes”. Mayor distancia implica mayor incertidumbre:
IDW no convierte una estimación en observación ni garantiza igual confiabilidad superficial.

La convención espacial es única: `x/column = longitud` (oeste→este) y `y/row = latitud`.
Las filas de cálculo avanzan sur→norte y son uniformes en Web Mercator, la proyección con la que
Leaflet estira un `ImageOverlay`; la resolución configurada limita su separación angular máxima.
El RGBA completo (color, máscara y alpha) se invierte verticalmente una sola vez al codificar el
PNG en `orient_rgba_for_leaflet`, por lo que la fila visible 0 es el norte. Los bounds enviados a
Leaflet son bordes exteriores de píxel en orden `[[south, west], [north, east]]`.

El HTML incorpora los datos meteorológicos y el GeoJSON y puede copiarse directamente a un
hosting estático. Folium/Leaflet y el mapa base usan recursos HTTPS externos; si el servidor de
teselas no está disponible, las capas propias continúan estando embebidas pero el fondo puede no
visualizarse.

## Estadísticas y auditoría

El panel trimestral muestra observaciones originales incluidas, estaciones, datasets, fuentes,
mínimo, máximo, media y mediana. Cada ejecución genera `output/auditoria.json` con fecha de
generación, resultados por dataset, descartes, duplicados, faltantes, cobertura temporal,
estadísticos, escala global, estaciones fuera del territorio y muestras trazables desde el XLS
hasta el acumulado publicado. El Parquet conserva precisión completa para revisión independiente.

## Rendimiento y controles

La representación web deduplica los metadatos de cada estación y conserva una sola referencia
compacta por estación y período. Los límites provinciales se simplifican únicamente para su
representación visual, preservando topología; la máscara IDW utiliza la geometría original. Estas
optimizaciones no modifican el Parquet ni la metodología científica. El mapa permite reproducir,
arrastrar el slider, usar los botones anterior/siguiente o navegar con las flechas del teclado.

## Análisis climático avanzado

La versión 1.1 incorpora una normal climática configurable por estación y trimestre. El período
predeterminado es 1991–2020 y se exigen al menos cinco años distintos; sin ese mínimo no se
calcula anomalía. La anomalía absoluta resta la normal al acumulado observado. La relativa divide
esa diferencia por la normal y no se calcula cuando la normal vale cero.

El panel “Análisis avanzado” permite alternar precipitación absoluta, anomalía en milímetros y
anomalía porcentual; cada modalidad utiliza su propia escala global. También permite filtrar por
provincia y fuente, seleccionar una estación, dibujar su serie y comparar un mismo trimestre
entre años. La capa “Distancia a observación” expresa soporte espacial, no probabilidad. El
archivo `output/evaluacion_idw.json` registra MAE y RMSE mediante validación
leave-one-station-out. RBF y Kriging permanecen como extensiones futuras hasta contar con una
comparación objetiva equivalente.

Los filtros modifican los puntos observados, las estadísticas, la serie y la capa de soporte;
la superficie IDW permanece calculada con la red nacional completa y se identifica como tal.
El límite inicial provisional de 350 km es configurable y evita extrapolaciones especialmente
extensas: no se presenta como un umbral universal y debe recalibrarse según la densidad,
distribución territorial y distancia típica entre estaciones. Las distancias se calculan en un
CRS métrico azimutal equidistante centrado en Argentina (`lat_0=-34`, `lon_0=-63`); EPSG:4326 se
conserva sólo para datos y visualización. En la red actual, frente a distancias geodésicas WGS84,
la aproximación presenta un error absoluto mediano de 0,024 %, percentil 95 de 0,183 % y máximo
observado de 1,565 %. Por ello, el corte de 350 km debe interpretarse con esa tolerancia. La interfaz de
interpolación acepta explícitamente un nombre de método, pero rechaza RBF o Kriging hasta que
se implementen y validen, en lugar de sustituir silenciosamente el algoritmo.

## Calidad

```bash
uv run pytest
uv run ruff check .
```

## Publicación estática

Ejecutar `uv run precipitaciones` y copiar `output/index.html` al directorio público de GitHub
Pages, GitLab Pages, Netlify, Vercel o cualquier servidor estático. No se debe publicar ni
ejecutar Python, `uv`, una base de datos o un backend. Para probar localmente:

```bash
uv run python -m http.server 8000 -d output
```

Abrir `http://localhost:8000/`. Los datos de precipitación, estaciones, períodos, estadísticas,
límites e interpolaciones están embebidos. Requieren Internet las bibliotecas servidas por CDN
(Leaflet, jQuery, Bootstrap y complementos de Folium), además de las teselas OpenStreetMap.

## Salidas

- `output/index.html`: aplicación estática interactiva.
- `output/datos_normalizados.parquet`: datos trimestrales auditables.
- `output/auditoria.json`: métricas, advertencias y trazabilidad de la ejecución.
- `output/evaluacion_idw.json`: validación cruzada espacial de IDW.

## Limitaciones

- **Interpolación:** IDW genera estimaciones; no constituye una medición real.
- **Cobertura:** la distribución espacial de estaciones es heterogénea.
- **Distancia:** la incertidumbre aumenta al alejarse de las observaciones.
- **Datos faltantes:** ausencia de dato no equivale a precipitación cero.
- **Calidad:** el resultado depende de la calidad y continuidad de los XLS originales.
- **Escala temporal:** los acumulados usan obligatoriamente agregación trimestral `sum`.

## Checklist de versión 1.0.0

- [x] `uv sync --frozen`, pytest y Ruff funcionan.
- [x] Catálogo, Parquet, auditoría, mapa, escala, IDW, cobertura y estadísticas validados.
- [x] HTML preparado para archivo local y servidor HTTP estático.
- [x] Todos los XLS declarados están disponibles y se procesan.
- [x] Auditoría sin datasets omitidos ni errores de carga.
