import pandas as pd
import matplotlib.pyplot as plt

# 1. Crear diccionario con datos de paquetería en tránsito
paqueteria_transito = {
    'id_guia': [111, 222, 333, 444, 555, 666],
    'remitente': ['Tienda A', 'Empresa B', 'Startup C', 'Tienda A', 'Particular', 'Empresa B'],
    'destinatario': ['Juan', 'Maria', 'Pedro', 'Ana', 'Luis', 'Sofia'],
    'origen': ['CDMX', 'Guadalajara', 'Monterrey', 'CDMX', 'Puebla', 'Guadalajara'],
    'destino': ['Puebla', 'CDMX', 'CDMX', 'Monterrey', 'CDMX', 'Monterrey'],
    'peso_kg': [2.5, 15.0, 1.2, 5.0, 0.5, 10.0],
    'fecha_envio': ['2024-01-10', '2024-01-08', '2024-01-11', '2024-01-09', '2024-01-12', '2024-01-07'],
    'fecha_entrega_estimada': ['2024-01-12', '2024-01-10', '2024-01-13', '2024-01-11', '2024-01-13', '2024-01-09'],
    'costo_envio': [150, 450, 120, 250, 90, 350],
    'estatus': ['En transito', 'Entregado', 'En transito', 'En transito', 'En reparto', 'Entregado']
}

print("Analisis Exploratorio de Datos EDA")
print("Exploratory Data Analysis por Nicole Robles NC 0120")
print("5. Crear el diccionario con 10 listas y 6 elementos")

# 2. Convertir diccionario en DataFrame
df = pd.DataFrame(paqueteria_transito)
print("\n--- TABLA ORIGINAL DE PAQUETERÍA EN TRÁNSITO ---")
print(df)
print("\n✅ Archivo CSV creado correctamente")
print("+-+-+ 8. Leer los datos desde el archivo CSV+-+-+")
print("+-+-+ paqueteria_transito_0120.csv+-+-+")
print("\nDICCIONARIO ML")

# 3. Exportar los datos a CSV
print("+-+-+ Convertir el DataFrame en archivo CSV+-+-+")
df.to_csv(
    'paqueteria_transito.csv',
    index=False,
    encoding='utf-8-sig'
)

# 4. Leer el archivo CSV
df = pd.read_csv('paqueteria_transito.csv')
print("\n--- DATOS LEÍDOS DESDE CSV ---")
print(df)
print("+-+-+9. Análisis Exploratorio de Datos (EDA)+-+-+") 
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 5. Primeros y últimos registros 
print("+-+-+ 1. PRIMEROS REGISTROS (Head): +-+-+") 
print("+-+-+ 2. ÚLTIMOS REGISTROS (Tail): +-+-+") 
print(pd.concat([df.head(3), df.tail(2)]))

# 6. Dimensiones y estructura 
print("+-+-+6. Dimensiones y estructura+-+-+")
print("\n3. DIMENSIONES (Filas, Columnas):", df.shape) 
print("\n4. NOMBRES DE COLUMNAS:\n", df.columns) 
print("\n5. TIPOS DE DATOS:\n", df.dtypes)
print("\n6. INFORMACIÓN GENERAL:")
df.info()

# 7. Estadísticas y limpieza
print("\n7. ESTADÍSTICAS DESCRIPTIVAS:")
print(df.describe())

print("\n8. VALORES FALTANTES (Nulls):")
print(df.isnull().sum())

print("\n9. REGISTROS DUPLICADOS:")
print(df.duplicated().sum())

print("\n10. VALORES ÚNICOS POR COLUMNA:")
print(df.nunique())
print("Estatus únicos:", df['estatus'].unique())

# 8. Análisis específico
print(df)
print("-------------------------------------------------------------------------------------------------------------------------")
print("\n11. PAQUETES POR ESTATUS:")
print(df['estatus'].value_counts())

print("\n12. PROMEDIO DE COSTO DE ENVÍO: $", df['costo_envio'].mean())
print("13. COSTO DE ENVÍO MÁXIMO: $", df['costo_envio'].max())
print("14. COSTO DE ENVÍO MÍNIMO: $", df['costo_envio'].min())

# 9. Ordenamiento, filtrado y agrupación
print("\n15. PAQUETES ORDENADOS POR COSTO DE ENVÍO (Mayor a menor):")
print(df.sort_values(by='costo_envio', ascending=False))

print("\n16. PAQUETES CON COSTO DE ENVÍO MAYOR A $200:")
filtro = df[df['costo_envio'] > 200]
print(filtro)

print("\n17. RESUMEN POR ESTATUS:")
resumen = df.groupby('estatus').agg(
    paquetes=('id_guia', 'count'),
    costo_promedio=('costo_envio', 'mean'),
    costo_maximo=('costo_envio', 'max')
)
print(resumen)

# 18. Gráfica de barras: Paquetes por estatus
conteo = df['estatus'].value_counts()
plt.figure(figsize=(8, 5))
plt.bar(conteo.index, conteo.values, color='skyblue')
plt.title('Paquetes por estatus EDA Nicole Robles NC 0120')
plt.xlabel('Estatus')
plt.ylabel('Cantidad de paquetes')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 19. Gráfica de barras: Costo de envío por guía
plt.figure(figsize=(8, 5))
plt.bar(df['id_guia'].astype(str), df['costo_envio'], color='lightgreen')
plt.title('Costo de envío por número de guía EDA Nicole Robles NC 0120')
plt.xlabel('ID Guía')
plt.ylabel('Costo de envío en pesos')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 20. Histograma: Distribución de pesos
plt.figure(figsize=(8, 5))
plt.hist(df['peso_kg'], bins=5, color='orange', edgecolor='black')
plt.title('Distribución de pesos de paquetes (kg) EDA Nicole Robles NC 0120')
plt.xlabel('Peso (kg)')
plt.ylabel('Frecuencia')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 21. Gráfica de dispersión: Peso vs Costo de envío
plt.figure(figsize=(8, 5))
plt.scatter(df['peso_kg'], df['costo_envio'], color='red', s=100)
plt.title('Relación entre peso (kg) y costo de envío EDA Nicole Robles NC 0120')
plt.xlabel('Peso (kg)')
plt.ylabel('Costo de envío')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 22. Diagrama de caja: Distribución de costos de envío
plt.figure(figsize=(7, 5))
plt.boxplot(df['costo_envio'], patch_artist=True, boxprops=dict(facecolor='lightblue'))
plt.title('Distribución de costos de envío (Boxplot) EDA Nicole Robles NC 0120')
plt.ylabel('Costo de envío')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 23. Matriz de correlación
correlacion = df.corr(numeric_only=True)
print("\n--- MATRIZ DE CORRELACIÓN ---")
print(correlacion)

plt.figure(figsize=(9, 6))
plt.imshow(correlacion, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(correlacion.columns)), correlacion.columns, rotation=45)
plt.yticks(range(len(correlacion.columns)), correlacion.columns)
plt.title('Matriz de correlación de variables numéricas EDA Nicole Robles NC 0120')
plt.tight_layout()
plt.show()