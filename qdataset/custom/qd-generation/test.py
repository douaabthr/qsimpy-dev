import pandas as pd

# Charger le fichier CSV (remplace par ton chemin)
file_path = r"D:\Study\Master\master2\semstre3\PFE\tools\qsimpy_dev\qsimpy\qdataset\custom\datasets\qdataset_1000_sub_26.csv"
df = pd.read_csv(file_path)

# Vérifier que la colonne existe
if "original_width" not in df.columns:
    print("La colonne 'original_width' n'existe pas dans le fichier.")
else:
    # Calcul des statistiques
    sup_65 = (df["original_width"] > 65).sum()
    inf_65 = (df["original_width"] <= 65).sum()
    total = len(df)

    # Affichage
    print(f"Nombre total d'instances : {total}")
    print(f"Nombre d'instances avec original_width > 65 : {sup_65}")
    print(f"Nombre d'instances avec original_width <= 65 : {inf_65}")

    # (Optionnel) pourcentage
    print(f"Pourcentage > 65 : {sup_65 / total * 100:.2f}%")
    print(f"Pourcentage <= 65 : {inf_65 / total * 100:.2f}%")