"""
Client Open Library API
Responsable : récupération de livres distants pour enrichir la recherche locale.
"""

from typing import Dict, List

import requests


class OpenLibraryAPI:
    """Client simple pour interroger Open Library."""

    def __init__(self, base_url: str = "https://openlibrary.org", timeout: int = 8):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.max_books = 200

    def _normaliser_description(self, valeur) -> str:
        """Normalise les différentes formes de description renvoyées par l'API."""
        if not valeur:
            return "Description non disponible"

        if isinstance(valeur, str):
            return valeur.strip() or "Description non disponible"

        if isinstance(valeur, dict):
            texte = valeur.get("value", "")
            if isinstance(texte, str) and texte.strip():
                return texte.strip()

        if isinstance(valeur, list):
            parties = []
            for item in valeur:
                if isinstance(item, str) and item.strip():
                    parties.append(item.strip())
                elif isinstance(item, dict) and isinstance(item.get("value"), str):
                    parties.append(item["value"].strip())
            if parties:
                return " ".join(parties)

        return "Description non disponible"

    def _charger_description_work(self, work_key: str) -> str:
        """Récupère la description d'une oeuvre via l'endpoint /works/<id>.json."""
        if not work_key:
            return "Description non disponible"

        try:
            url = f"{self.base_url}{work_key}.json"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return self._normaliser_description(data.get("description"))
        except requests.RequestException:
            return "Description non disponible"

    def rechercher_livres(self, requete: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        Recherche des livres dans Open Library.

        Returns:
            List[Dict[str, str]]: Liste normalisée de livres avec les clés
            id, titre, auteur, genre, description, source_url.
        """
        if not requete or not requete.strip():
            return []

        limite_reelle = max(1, min(limit, self.max_books))

        params = {
            "q": requete.strip(),
            "limit": limite_reelle,
            "fields": "key,title,author_name,subject,first_sentence,language",
        }

        try:
            response = self.session.get(f"{self.base_url}/search.json?page=20", params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException:
            return []

        livres = []
        docs = payload.get("docs", [])
        nb_fallback_work = 0
        limite_fallback_work = min(max(limite_reelle // 4, 10), 30)

        for doc in docs:
            work_key = doc.get("key", "")
            titre = (doc.get("title") or "Titre inconnu").strip()

            auteurs = doc.get("author_name") or []
            auteur = ", ".join(auteurs[:2]) if auteurs else "Auteur inconnu"

            sujets = doc.get("subject") or []
            genre = ", ".join(sujets[:3]) if sujets else "Genre non spécifié"

            description = self._normaliser_description(doc.get("first_sentence"))
            if description == "Description non disponible" and work_key and nb_fallback_work < limite_fallback_work:
                description = self._charger_description_work(work_key)
                nb_fallback_work += 1

            livres.append(
                {
                    "id": len(livres) + 1,
                    "work_key": work_key,
                    "titre": titre,
                    "auteur": auteur,
                    "genre": genre,
                    "description": description,
                    "source_url": f"{self.base_url}{work_key}" if work_key else self.base_url,
                }
            )

            if len(livres) >= limite_reelle:
                break

        return livres
