
"""
Module de recherche
Responsable : traitement des requêtes, calcul de pertinence, recherche
"""

import sys
import os
import time
from typing import List, Dict, Tuple, Optional

# Ajout du dossier parent pour permettre les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.preprocesseur import Preprocesseur
from modules.indexeur import Indexeur


class MoteurRecherche:
    """
    Classe qui gère la recherche dans les livres indexés
    """
    
    def __init__(self, indexeur: Indexeur):
        """
        Initialise le moteur de recherche avec un indexeur existant
        """
        self.indexeur = indexeur
        self.preprocesseur = indexeur.preprocesseur
        
        print(f"✅ Moteur de recherche initialisé avec {self.indexeur.nombre_livres()} livres")
    
    # ================================================================
    # ÉTAPE 4.1 : PRÉTRAITEMENT DE LA REQUÊTE
    # ================================================================
    
    def pretraiter_requete(self, requete: str) -> List[str]:
        """
        ÉTAPE 4.1 : Applique le même prétraitement que pour les livres
        Étapes :
        1. Nettoyage (minuscules, accents, ponctuation, chiffres)
        2. Tokenisation (découpage en mots)
        3. Suppression des stop words
        
        Args:
            requete (str): La requête utilisateur brute
            
        Returns:
            List[str]: Liste des tokens de la requête (mots utiles)
        """
        if not requete or not isinstance(requete, str):
            return []
        
        # Utilisation directe de la méthode tokenizer du préprocesseur
        # qui fait déjà : nettoyage → tokenisation → suppression stop words
        tokens = self.preprocesseur.tokenizer(requete)
        
        return tokens
    
    def afficher_pretraitement_requete(self, requete: str):
        """
        Affiche les étapes de prétraitement d'une requête (utile pour déboguer)
        
        Args:
            requete (str): La requête utilisateur brute
        """
        print("\n" + "=" * 60)
        print("🔧 ÉTAPE 4.1 : PRÉTRAITEMENT DE LA REQUÊTE")
        print("=" * 60)
        print(f"  📝 Requête originale : {requete}")
        
        # Étape 1 : Nettoyage
        texte_nettoye = self.preprocesseur.nettoyer(requete)
        print(f"  🧹 Après nettoyage   : {texte_nettoye}")
        
        # Étape 2 : Tokenisation (sans filtre)
        tokens_bruts = self.preprocesseur.tokeniser(texte_nettoye)
        print(f"  ✂️ Tokens bruts      : {tokens_bruts}")
        
        # Étape 3 : Suppression des stop words
        tokens_filtres = self.preprocesseur.filtrer_stop_words(tokens_bruts)
        print(f"  🚫 Après stop words  : {tokens_filtres}")
        
        print("-" * 60)
        print(f"  ✅ Requête traitée : {' '.join(tokens_filtres)}")
        print("=" * 60)
        
        return tokens_filtres
    
    # ================================================================
    # ÉTAPE 4.2 : CALCUL DU SCORE DE PERTINENCE
    # ================================================================
    
    def calculer_score_pertinence(self, requete_tokens: List[str], livre_id: int) -> float:
        """
        Calcule le score de pertinence d'un livre par rapport à une requête
        Formule : Score = somme des TF-IDF de chaque token de la requête dans le livre
        
        Args:
            requete_tokens (List[str]): Tokens de la requête prétraitée
            livre_id (int): ID du livre à évaluer
            
        Returns:
            float: Score de pertinence (plus le score est élevé, plus le livre est pertinent)
        """
        score = 0.0
        
        for token in requete_tokens:
            # Récupérer le TF-IDF du token dans ce livre
            tfidf = self.indexeur.obtenir_tfidf(livre_id, token)
            score += tfidf
        
        return score
    
    def rechercher(self, requete: str, top_n: int = 10) -> List[Tuple[Dict, float]]:
        """
        Recherche les livres les plus pertinents pour une requête
        
        Args:
            requete (str): La requête utilisateur
            top_n (int): Nombre maximum de résultats à retourner
            
        Returns:
            List[Tuple[Dict, float]]: Liste de tuples (livre, score) triés par score décroissant
        """
        # Étape 4.1 : Prétraitement de la requête
        requete_tokens = self.pretraiter_requete(requete)
        
        if not requete_tokens:
            print("⚠️ Aucun terme de recherche valide après prétraitement")
            return []
        
        # Calcul du score pour chaque livre
        scores = []
        for livre_id in self.indexeur.obtenir_tous_les_ids():
            score = self.calculer_score_pertinence(requete_tokens, livre_id)
            if score > 0:  # On garde seulement les livres avec un score > 0
                livre = self.indexeur.obtenir_livre(livre_id)
                if livre:
                    scores.append((livre, score))
        
        # Tri par score décroissant
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Limitation au top N
        return scores[:top_n]
    
    def rechercher_avec_details(self, requete: str, top_n: int = 10) -> Dict:
        """
        Recherche avec détails supplémentaires (temps, tokens, etc.)
        
        Args:
            requete (str): La requête utilisateur
            top_n (int): Nombre maximum de résultats
            
        Returns:
            Dict: Dictionnaire contenant les résultats et les métadonnées
        """
        # Mesurer le temps de recherche
        debut = time.time()
        
        # Prétraitement
        requete_tokens = self.pretraiter_requete(requete)
        
        # Recherche
        resultats = self.rechercher(requete, top_n)
        
        # Calcul du temps écoulé
        temps_ms = (time.time() - debut) * 1000
        
        return {
            "requete_originale": requete,
            "requete_tokens": requete_tokens,
            "requete_traitee": " ".join(requete_tokens),
            "nb_resultats": len(resultats),
            "temps_ms": temps_ms,
            "resultats": resultats
        }
    
    # ================================================================
    # AFFICHAGE DES RÉSULTATS
    # ================================================================
    
    def afficher_resultats(self, requete: str, top_n: int = 10):
        """
        Affiche les résultats de recherche de manière lisible
        
        Args:
            requete (str): La requête utilisateur
            top_n (int): Nombre maximum de résultats à afficher
        """
        print("\n" + "=" * 70)
        print("🔍 RECHERCHE DE LIVRES")
        print("=" * 70)
        print(f"  Requête : {requete}")
        
        details = self.rechercher_avec_details(requete, top_n)
        
        print(f"\n  📊 Résultats : {details['nb_resultats']} livre(s) trouvé(s) en {details['temps_ms']:.2f} ms")
        print(f"  🔤 Requête traitée : {details['requete_traitee']}")
        
        if not details['resultats']:
            print("\n  ❌ Aucun résultat trouvé.")
            print("\n  💡 Suggestions :")
            print("     - Vérifiez l'orthographe des mots")
            print("     - Essayez des mots plus génériques")
            print("     - Utilisez des mots-clés plus courts")
            return
        
        print("\n" + "-" * 70)
        print("  📖 RÉSULTATS (du plus pertinent au moins pertinent) :")
        print("-" * 70)
        
        for i, (livre, score) in enumerate(details['resultats'], 1):
            titre = livre.get('titre', 'Titre inconnu')
            auteur = livre.get('auteur', 'Auteur inconnu')
            genre = livre.get('genre', 'Genre inconnu')
            description = livre.get('description', '')
            description_courte = description[:150] + "..." if len(description) > 150 else description
            
            # Création d'une barre de pertinence visuelle
            max_score = details['resultats'][0][1] if details['resultats'] else 1
            barre_longueur = int((score / max_score) * 30) if max_score > 0 else 0
            barre = "█" * barre_longueur
            
            print(f"\n  {i}. 📚 {titre}")
            print(f"     ✍️  Auteur : {auteur}")
            print(f"     🏷️  Genre : {genre}")
            print(f"     📝 Description : {description_courte}")
            print(f"     ⭐ Score : {score:.6f} {barre}")
        
        print("\n" + "=" * 70)
    
    def afficher_resultats_simple(self, requete: str, top_n: int = 10):
        """
        Affiche les résultats de manière simplifiée (juste titres et scores)
        
        Args:
            requete (str): La requête utilisateur
            top_n (int): Nombre maximum de résultats à afficher
        """
        resultats = self.rechercher(requete, top_n)
        
        if not resultats:
            print(f"\n  Aucun résultat trouvé pour : '{requete}'")
            return
        
        print(f"\n  {len(resultats)} résultat(s) pour '{requete}':")
        for i, (livre, score) in enumerate(resultats, 1):
            titre = livre.get('titre', 'Inconnu')
            print(f"    {i}. {titre} (score: {score:.6f})")
    
    # ================================================================
    # RECHERCHE AVANCÉE
    # ================================================================
    
    def rechercher_par_genre(self, genre: str, top_n: int = 10) -> List[Tuple[Dict, float]]:
        """
        Recherche des livres par genre (filtre avant recherche)
        
        Args:
            genre (str): Genre recherché
            top_n (int): Nombre maximum de résultats
            
        Returns:
            List[Tuple[Dict, float]]: Liste des livres du genre avec leur score
        """
        genre_tokens = self.pretraiter_requete(genre)
        
        if not genre_tokens:
            return []
        
        scores = []
        for livre in self.indexeur.obtenir_tous_les_livres():
            livre_genre = livre.get('genre', '')
            livre_genre_tokens = self.pretraiter_requete(livre_genre)
            
            # Calcul du score de correspondance avec le genre
            score = 0
            for token in genre_tokens:
                if token in livre_genre_tokens:
                    score += 1
            
            if score > 0:
                scores.append((livre, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]
    
    def rechercher_par_auteur(self, auteur: str, top_n: int = 10) -> List[Tuple[Dict, float]]:
        """
        Recherche des livres par auteur
        
        Args:
            auteur (str): Nom de l'auteur recherché
            top_n (int): Nombre maximum de résultats
            
        Returns:
            List[Tuple[Dict, float]]: Liste des livres de l'auteur
        """
        auteur_tokens = self.pretraiter_requete(auteur)
        
        if not auteur_tokens:
            return []
        
        scores = []
        for livre in self.indexeur.obtenir_tous_les_livres():
            livre_auteur = livre.get('auteur', '')
            livre_auteur_tokens = self.pretraiter_requete(livre_auteur)
            
            score = 0
            for token in auteur_tokens:
                if token in livre_auteur_tokens:
                    score += 1
            
            if score > 0:
                scores.append((livre, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]
    
    # ================================================================
    # STATISTIQUES DE RECHERCHE
    # ================================================================
    
    def get_mots_cles_importants(self, livre_id: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Retourne les mots les plus importants (TF-IDF élevé) d'un livre
        
        Args:
            livre_id (int): ID du livre
            top_n (int): Nombre de mots à retourner
            
        Returns:
            List[Tuple[str, float]]: Liste des (mot, score) triés par score décroissant
        """
        if not hasattr(self.indexeur, 'tfidf_matrix'):
            return []
        
        if livre_id not in self.indexeur.tfidf_matrix:
            return []
        
        tfidf_dict = self.indexeur.tfidf_matrix[livre_id]
        mots_tries = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return mots_tries


# ================================================================
# TESTS (ÉTAPE 4.1 ET 4.2)
# ================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧪 TEST DU MODULE RECHERCHE (ÉTAPES 4.1 ET 4.2)")
    print("=" * 70)
    
    # Création de l'indexeur
    print("\n📚 Chargement de l'indexeur...")
    indexeur = Indexeur(None, "data/stop_words_fr.txt")
    
    # Création du moteur de recherche
    moteur = MoteurRecherche(indexeur)
    
    # Test 1 : Prétraitement d'une requête (Étape 4.1)
    print("\n" + "=" * 70)
    print("🧪 TEST 1 : PRÉTRAITEMENT D'UNE REQUÊTE (Étape 4.1)")
    print("=" * 70)
    
    test_requetes = [
        "Le Petit Prince",
        "Je cherche un livre sur la mer",
        "Victor Hugo Les Misérables",
        "histoire d'amour et de justice",
        "aventure sous la mer"
    ]
    
    for requete in test_requetes:
        moteur.afficher_pretraitement_requete(requete)
    
    # Test 2 : Recherche simple
    print("\n" + "=" * 70)
    print("🧪 TEST 2 : RECHERCHE DE LIVRES")
    print("=" * 70)
    
    moteur.afficher_resultats("petit prince", top_n=5)
    
    # Test 3 : Recherche par auteur
    print("\n" + "=" * 70)
    print("🧪 TEST 3 : RECHERCHE PAR AUTEUR")
    print("=" * 70)
    
    resultats_auteur = moteur.rechercher_par_auteur("Victor Hugo", top_n=5)
    print(f"\n  Livres de Victor Hugo trouvés : {len(resultats_auteur)}")
    for livre, score in resultats_auteur:
        print(f"    - {livre.get('titre')} (score: {score:.4f})")
    
    # Test 4 : Recherche par genre
    print("\n" + "=" * 70)
    print("🧪 TEST 4 : RECHERCHE PAR GENRE")
    print("=" * 70)
    
    resultats_genre = moteur.rechercher_par_genre("Philosophique", top_n=5)
    print(f"\n  Livres de genre 'Philosophique' trouvés : {len(resultats_genre)}")
    for livre, score in resultats_genre:
        print(f"    - {livre.get('titre')} (score: {score:.4f})")
    
    # Test 5 : Mots clés importants d'un livre
    print("\n" + "=" * 70)
    print("🧪 TEST 5 : MOTS CLÉS IMPORTANTS D'UN LIVRE")
    print("=" * 70)
    
    mots_importants = moteur.get_mots_cles_importants(1, top_n=10)
    print("\n  Mots clés du 'Le Petit Prince' :")
    for mot, score in mots_importants:
        barre = "█" * int(score * 100)
        print(f"    {mot:15s} : {score:.6f} {barre}")
    
    # Test 6 : Recherche avancée
    print("\n" + "=" * 70)
    print("🧪 TEST 6 : RECHERCHE AVANCÉE")
    print("=" * 70)
    
    recherche_avancee = moteur.rechercher_avec_details("amour et aventure", top_n=5)
    print(f"\n  Résultats détaillés :")
    print(f"    Requête traitée : {recherche_avancee['requete_traitee']}")
    print(f"    Temps : {recherche_avancee['temps_ms']:.2f} ms")
    print(f"    Nombre de résultats : {recherche_avancee['nb_resultats']}")
    
    print("\n" + "=" * 70)
    print("✅ ÉTAPES 4.1 ET 4.2 TERMINÉES AVEC SUCCÈS !")
    print("=" * 70)