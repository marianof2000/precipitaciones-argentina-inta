# Precipitaciones Argentina

Proyecto Python para integrar observaciones de precipitación de estaciones argentinas y,
por etapas, construir un mapa geoespacial y temporal publicable como sitio estático.

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

## Procesamiento

La Etapa 1 lee y valida el catálogo, comprueba archivo/hoja/columnas, normaliza fechas,
coordenadas y unidades, descarta explícitamente registros inválidos o duplicados, y agrega por
estación y trimestre. El modelo conserva dataset, archivo, fuente, estación, ubicación, fecha,
año, trimestre, período, valor/unidad original y milímetros.

Los trimestres son T1 (enero-marzo), T2 (abril-junio), T3 (julio-septiembre) y T4
(octubre-diciembre). La agregación inicial es precipitación acumulada (`sum`) y es configurable;
un dato ausente nunca se reemplaza por cero.

```bash
uv run precipitaciones
```

La Etapa 1 genera `output/datos_normalizados.parquet`, con los datos trimestrales utilizados por
las siguientes etapas. La Etapa 2 genera `output/index.html`: incluye límites provinciales,
observaciones reales, cobertura de estaciones, selector de los 303 períodos, estadísticas y una
escala cromática global desde 0 hasta el máximo real, con cortes conceptuales cada 10 mm.

La Etapa 3 incorpora superficies IDW trimestrales. La grilla y su resolución se configuran en
`config.py`; cada superficie se restringe al territorio, al convex hull de las estaciones del
período y a una distancia máxima de 350 km respecto de una observación. Con menos de tres
ubicaciones no colineales, la estimación queda transparente como “Sin datos suficientes”. Las
estaciones “Sin asignar” se conservan como observaciones para trazabilidad, pero se excluyen de
IDW y cobertura. Mayor distancia a estaciones implica mayor incertidumbre: IDW no convierte una
estimación en observación ni garantiza igual confiabilidad en toda la superficie.

El HTML incorpora los datos meteorológicos y el GeoJSON y puede copiarse directamente a un
hosting estático. Folium/Leaflet y el mapa base usan recursos HTTPS externos; si el servidor de
teselas no está disponible, las capas propias continúan estando embebidas pero el fondo puede no
visualizarse.

## Rendimiento y controles

La representación web deduplica los metadatos de cada estación y conserva una sola referencia
compacta por estación y período. Los límites provinciales se simplifican únicamente para su
representación visual, preservando topología; la máscara IDW utiliza la geometría original. Estas
optimizaciones no modifican el Parquet ni la metodología científica. El mapa permite reproducir,
arrastrar el slider, usar los botones anterior/siguiente o navegar con las flechas del teclado.

## Calidad

```bash
uv run pytest
uv run ruff check .
```

## Limitaciones científicas

La interpolación futura será una estimación, nunca una observación. La cobertura es desigual y
la incertidumbre aumenta con la distancia a estaciones. La calidad final está condicionada por
los archivos originales. Ausencia de medición no significa precipitación cero.
