import csv

# Ouvre ton fichier log brut
with open('data/output_0.1.log', 'r', encoding='utf-8') as infile, open('data/dataset.csv', 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['texte', 'label'])  # En-têtes

    for line in infile:
        line = line.strip()
        if not line:
            continue

        # Exemple : suppose que le label est au début de chaque ligne, séparé par un espace
        # Modifie cette logique selon le vrai format du log
        try:
            parts = line.split(' ', 1)
            label = parts[0]
            texte = parts[1]
            writer.writerow([texte, label])
        except IndexError:
            continue  # saute les lignes mal formées
