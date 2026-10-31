"""Donnees de reference partagees (generation ET controles de coherence)."""

# Vraies communes (code postal, ville) : indispensable pour tester la coherence CP / ville.
# Les grandes villes pesent plus lourd, comme dans un vrai fichier clients.
COMMUNES = [
    ("75001", "Paris", 3), ("75011", "Paris", 4), ("75015", "Paris", 4), ("75018", "Paris", 3),
    ("75020", "Paris", 3), ("13001", "Marseille", 3), ("13008", "Marseille", 2), ("69001", "Lyon", 2),
    ("69003", "Lyon", 3), ("69007", "Lyon", 2), ("31000", "Toulouse", 3), ("06000", "Nice", 2),
    ("44000", "Nantes", 3), ("67000", "Strasbourg", 2), ("34000", "Montpellier", 2), ("33000", "Bordeaux", 3),
    ("59000", "Lille", 3), ("35000", "Rennes", 2), ("51100", "Reims", 1), ("76600", "Le Havre", 1),
    ("42000", "Saint-Étienne", 1), ("83000", "Toulon", 1), ("38000", "Grenoble", 2), ("21000", "Dijon", 1),
    ("49000", "Angers", 1), ("30000", "Nîmes", 1), ("63000", "Clermont-Ferrand", 1), ("72000", "Le Mans", 1),
    ("13100", "Aix-en-Provence", 1), ("29200", "Brest", 1), ("37000", "Tours", 1), ("80000", "Amiens", 1),
    ("87000", "Limoges", 1), ("74000", "Annecy", 1), ("66000", "Perpignan", 1), ("57000", "Metz", 1),
    ("25000", "Besançon", 1), ("45000", "Orléans", 1), ("76000", "Rouen", 1), ("14000", "Caen", 1),
    ("68100", "Mulhouse", 1), ("54000", "Nancy", 1), ("92100", "Boulogne-Billancourt", 2),
    ("93100", "Montreuil", 2), ("94000", "Créteil", 1), ("92000", "Nanterre", 1), ("93200", "Saint-Denis", 2),
    ("95000", "Cergy", 1), ("78000", "Versailles", 1), ("91000", "Évry-Courcouronnes", 1),
    ("77000", "Melun", 1), ("64000", "Pau", 1), ("17000", "La Rochelle", 1), ("86000", "Poitiers", 1),
    ("56100", "Lorient", 1), ("97400", "Saint-Denis", 1), ("2A000", "Ajaccio", 1),
]

# Index CP -> ville, pour le controle de coherence
VILLE_PAR_CP: dict[str, str] = {cp: ville for cp, ville, _ in COMMUNES}


# Prenoms connus, avec leur genre.
# Ici on reutilise la liste de Faker pour la demo. En production, on prendrait le
# "Fichier des prenoms" de l'INSEE (open data, environ 30 000 prenoms) : meme code, autre fichier.
def _prenoms() -> tuple[set[str], set[str]]:
    from faker.providers.person.fr_FR import Provider
    import unicodedata

    def cle(p: str) -> str:  # "Élisabeth" -> "ELISABETH" : on compare sans accents
        return "".join(c for c in unicodedata.normalize("NFD", p) if unicodedata.category(c) != "Mn").upper()

    return {cle(p) for p in Provider.first_names_female}, {cle(p) for p in Provider.first_names_male}


PRENOMS_FEMININS, PRENOMS_MASCULINS = _prenoms()
PRENOMS = PRENOMS_FEMININS | PRENOMS_MASCULINS
