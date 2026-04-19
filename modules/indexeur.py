"""
Module d'indexation des livres
Responsable : création de l'index, calcul du TF, calcul de l'IDF, calcul du TF-IDF

Étapes implémentées :
- Étape 3.1 : Structure de l'index (dictionnaire id_livre → tokens)
- Étape 3.2 : Calcul du TF (Term Frequency)
- Étape 3.3 : Calcul de l'IDF (Inverse Document Frequency)
- Étape 3.4 : Calcul du TF-IDF (Term Frequency × Inverse Document Frequency)
"""

import sys
import os
import json
import math
from collections import Counter
from typing import List, Dict, Set, Tuple, Optional

# Ajout du dossier parent pour permettre les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.preprocesseur import Preprocesseur


class Indexeur:
    """
    Classe qui gère l'indexation des livres
    Structure : dictionnaire {id_livre: [liste_des_tokens]}
    """
    
    def __init__(self, chemin_livres: str = "data/livres.json", 
                 chemin_stop_words: str = "data/stop_words_fr.txt"):
        """
        Initialise l'indexeur et charge les livres
        """
        self.chemin_livres = chemin_livres
        self.preprocesseur = Preprocesseur(chemin_stop_words)
        
        # Étape 3.1 : Structure de l'index
        self.index: Dict[int, List[str]] = {}
        
        # Étape 3.2 : Stockage des scores TF
        self.tf_scores: Dict[int, Dict[str, float]] = {}
        
        # Étape 3.3 : Stockage des scores IDF
        self.idf_scores: Dict[str, float] = {}
        self._mot_frequence_document: Dict[str, int] = {}
        
        # Étape 3.4 : Stockage des scores TF-IDF
        self.tfidf_matrix: Dict[int, Dict[str, float]] = {}
        
        # Stockage des livres complets
        self.livres: List[Dict] = []
        
        # Chargement des livres
        self.charger_livres()
        
        # Indexation automatique
        self.indexer_tous_les_livres()
        
        # Calcul automatique du TF
        self.calculer_tf_pour_tous_les_livres()
        
        # Calcul automatique de l'IDF
        self.calculer_idf_et_stocker()
        
        # Calcul automatique du TF-IDF
        self.calculer_tfidf_et_stocker()
    
    # ================================================================
    # CHARGEMENT DES LIVRES DEPUIS LE FICHIER JSON
    # ================================================================
    
    def charger_livres(self) -> List[Dict]:
        """Charge les livres depuis le fichier JSON"""
        try:
            with open(self.chemin_livres, 'r', encoding='utf-8') as fichier:
                self.livres = json.load(fichier)
            print(f"✅ {len(self.livres)} livres chargés depuis {self.chemin_livres}")
            return self.livres
        except FileNotFoundError:
            print(f"❌ Fichier {self.chemin_livres} non trouvé !")
            self.livres = []
            return []
        except json.JSONDecodeError:
            print(f"❌ Erreur de lecture du fichier JSON {self.chemin_livres}")
            self.livres = []
            return []
    
    def sauvegarder_livres(self):
        """Sauvegarde les livres dans le fichier JSON"""
        try:
            with open(self.chemin_livres, 'w', encoding='utf-8') as fichier:
                json.dump(self.livres, fichier, ensure_ascii=False, indent=4)
            print(f"✅ {len(self.livres)} livres sauvegardés")
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde : {e}")
    
    # ================================================================
    # EXTRACTION DU TEXTE À INDEXER
    # ================================================================
    
    def extraire_texte_a_indexer(self, livre: Dict) -> str:
        """Combine titre, description, auteur et genre"""
        titre = livre.get('titre', '')
        description = livre.get('description', '')
        auteur = livre.get('auteur', '')
        genre = livre.get('genre', '')
        return f"{titre} {description} {auteur} {genre}"
    
    # ================================================================
    # ÉTAPE 3.1 : INDEXATION
    # ================================================================
    
    def indexer_livre(self, livre: Dict) -> List[str]:
        """Indexe un seul livre et retourne ses tokens"""
        texte = self.extraire_texte_a_indexer(livre)
        return self.preprocesseur.tokenizer(texte)
    
    def indexer_tous_les_livres(self):
        """Indexe tous les livres"""
        print("\n" + "=" * 60)
        print("📚 ÉTAPE 3.1 : INDEXATION DES LIVRES")
        print("=" * 60)
        
        self.index = {}
        
        for livre in self.livres:
            livre_id = livre.get('id')
            if livre_id is None:
                continue
            
            tokens = self.indexer_livre(livre)
            self.index[livre_id] = tokens
            
            titre = livre.get('titre', 'Inconnu')
            print(f"  📖 [{livre_id}] {titre[:30]:30s} → {len(tokens)} tokens")
        
        print("-" * 60)
        print(f"✅ {len(self.index)} livres indexés")
        print("=" * 60)
    
    # ================================================================
    # ÉTAPE 3.2 : CALCUL DU TF
    # ================================================================
    
    def calculer_tf(self, tokens: List[str]) -> Dict[str, float]:
        """
        Calcule la Term Frequency (TF) pour une liste de tokens
        Formule : TF = (nb occurrences) / (nombre total de mots)
        """
        if not tokens:
            return {}
        
        compteur = Counter(tokens)
        longueur = len(tokens)
        
        return {mot: occurrence / longueur for mot, occurrence in compteur.items()}
    
    def calculer_tf_pour_tous_les_livres(self):
        """Calcule le TF pour tous les livres indexés"""
        print("\n" + "=" * 60)
        print("📊 ÉTAPE 3.2 : CALCUL DU TF (TERM FREQUENCY)")
        print("=" * 60)
        
        self.tf_scores = {}
        
        for livre_id, tokens in self.index.items():
            tf = self.calculer_tf(tokens)
            self.tf_scores[livre_id] = tf
            
            livre = self.obtenir_livre(livre_id)
            titre = livre.get('titre', 'Inconnu') if livre else 'Inconnu'
            
            if tf:
                top_mots = sorted(tf.items(), key=lambda x: x[1], reverse=True)[:3]
                top_affichage = ', '.join([f"{mot}({tf:.3f})" for mot, tf in top_mots])
                print(f"  📖 [{livre_id}] {titre[:25]:25s} → Top TF : {top_affichage}")
        
        print("-" * 60)
        print(f"✅ TF calculé pour {len(self.tf_scores)} livres")
        print("=" * 60)
    
    def obtenir_tf(self, livre_id: int, mot: str) -> float:
        """Récupère le TF d'un mot spécifique dans un livre"""
        if livre_id not in self.tf_scores:
            return 0.0
        return self.tf_scores[livre_id].get(mot, 0.0)
    
    def afficher_tf_pour_livre(self, livre_id: int, limite: int = 10):
        """Affiche les scores TF pour un livre spécifique"""
        if livre_id not in self.tf_scores:
            print(f"❌ Livre {livre_id} non trouvé")
            return
        
        livre = self.obtenir_livre(livre_id)
        titre = livre.get('titre', 'Inconnu') if livre else 'Inconnu'
        tf_dict = self.tf_scores[livre_id]
        
        print(f"\n📖 TF pour : {titre} (ID: {livre_id})")
        print("-" * 40)
        
        mots_tries = sorted(tf_dict.items(), key=lambda x: x[1], reverse=True)[:limite]
        
        for mot, tf in mots_tries:
            barre = "█" * int(tf * 50)
            print(f"  {mot:15s} : {tf:.4f} {barre}")
    
    # ================================================================
    # ÉTAPE 3.3 : CALCUL DE L'IDF
    # ================================================================
    
    def calculer_idf(self) -> Dict[str, float]:
        """
        Calcule l'Inverse Document Frequency (IDF) pour chaque mot
        Formule : IDF(mot) = log( N / (1 + n) )
        """
        if not self.index:
            return {}
        
        N = len(self.index)
        
        # Compter dans combien de livres apparaît chaque mot
        self._mot_frequence_document = {}
        
        for tokens in self.index.values():
            mots_uniques = set(tokens)
            for mot in mots_uniques:
                self._mot_frequence_document[mot] = self._mot_frequence_document.get(mot, 0) + 1
        
        # Calculer l'IDF pour chaque mot
        idf_scores = {}
        for mot, doc_count in self._mot_frequence_document.items():
            idf = math.log(N / (1 + doc_count))
            idf_scores[mot] = idf
        
        return idf_scores
    
    def calculer_idf_et_stocker(self):
        """Calcule l'IDF pour tous les mots et stocke les résultats"""
        print("\n" + "=" * 60)
        print("📊 ÉTAPE 3.3 : CALCUL DE L'IDF (INVERSE DOCUMENT FREQUENCY)")
        print("=" * 60)
        
        self.idf_scores = self.calculer_idf()
        
        print(f"  📚 Nombre total de livres : {len(self.index)}")
        print(f"  🔤 Nombre de mots uniques : {len(self.idf_scores)}")
        
        if self.idf_scores:
            mots_rares = sorted(self.idf_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            print("\n  🦄 Mots les plus rares (IDF élevé) :")
            for mot, idf in mots_rares:
                print(f"     {mot:15s} : IDF = {idf:.4f}")
            
            mots_frequents = sorted(self.idf_scores.items(), key=lambda x: x[1])[:10]
            print("\n  📖 Mots les plus fréquents (IDF faible) :")
            for mot, idf in mots_frequents:
                print(f"     {mot:15s} : IDF = {idf:.4f}")
        
        print("-" * 60)
        print(f"✅ IDF calculé pour {len(self.idf_scores)} mots uniques")
        print("=" * 60)
    
    def obtenir_idf(self, mot: str) -> float:
        """Récupère l'IDF d'un mot spécifique"""
        if not hasattr(self, 'idf_scores'):
            return 0.0
        return self.idf_scores.get(mot, 0.0)
    
    def obtenir_frequence_document(self, mot: str) -> int:
        """Récupère le nombre de livres contenant un mot"""
        if not hasattr(self, '_mot_frequence_document'):
            return 0
        return self._mot_frequence_document.get(mot, 0)
    
    def afficher_idf_pour_mots(self, mots: List[str]):
        """Affiche les scores IDF pour une liste de mots"""
        print("\n📊 Scores IDF pour des mots spécifiques :")
        print("-" * 40)
        
        for mot in mots:
            idf = self.obtenir_idf(mot)
            frequence = self.obtenir_frequence_document(mot)
            print(f"  '{mot}' : IDF = {idf:.4f} (apparaît dans {frequence} livre(s))")
    
    # ================================================================
    # ÉTAPE 3.4 : CALCUL DU TF-IDF
    # ================================================================
    
    def calculer_tfidf(self) -> Dict[int, Dict[str, float]]:
        """
        ÉTAPE 3.4 : Calcule le TF-IDF pour chaque livre et chaque mot
        Formule : TF-IDF(mot, livre) = TF(mot, livre) × IDF(mot)
        
        Returns:
            Dict[int, Dict[str, float]]: Matrice {id_livre: {mot: tfidf}}
        """
        if not self.tf_scores or not self.idf_scores:
            print("❌ Erreur : TF ou IDF non calculé. Veuillez d'abord exécuter les étapes 3.2 et 3.3.")
            return {}
        
        tfidf_matrix = {}
        
        for livre_id, tf_dict in self.tf_scores.items():
            tfidf_dict = {}
            for mot, tf in tf_dict.items():
                idf = self.idf_scores.get(mot, 0)
                tfidf_dict[mot] = tf * idf
            tfidf_matrix[livre_id] = tfidf_dict
        
        return tfidf_matrix
    
    def calculer_tfidf_et_stocker(self):
        """ÉTAPE 3.4 : Calcule le TF-IDF pour tous les livres et stocke les résultats"""
        print("\n" + "=" * 60)
        print("📊 ÉTAPE 3.4 : CALCUL DU TF-IDF")
        print("=" * 60)
        
        self.tfidf_matrix = self.calculer_tfidf()
        
        if not self.tfidf_matrix:
            print("❌ Erreur lors du calcul du TF-IDF")
            return
        
        print(f"  📚 Nombre de livres : {len(self.tfidf_matrix)}")
        
        print("\n  📖 Top mots par livre (TF-IDF le plus élevé) :")
        print("-" * 50)
        
        for livre_id, tfidf_dict in self.tfidf_matrix.items():
            if tfidf_dict:
                top_mots = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:3]
                top_affichage = ', '.join([f"{mot}({score:.4f})" for mot, score in top_mots])
                
                livre = self.obtenir_livre(livre_id)
                titre = livre.get('titre', 'Inconnu') if livre else 'Inconnu'
                print(f"     [{livre_id}] {titre[:25]:25s} → {top_affichage}")
        
        print("-" * 60)
        print(f"✅ TF-IDF calculé pour {len(self.tfidf_matrix)} livres")
        print("=" * 60)
    
    def obtenir_tfidf(self, livre_id: int, mot: str) -> float:
        """
        Récupère le score TF-IDF d'un mot spécifique dans un livre
        
        Args:
            livre_id (int): ID du livre
            mot (str): Le mot recherché
            
        Returns:
            float: Score TF-IDF (0 si mot non trouvé)
        """
        if not hasattr(self, 'tfidf_matrix'):
            return 0.0
        if livre_id not in self.tfidf_matrix:
            return 0.0
        return self.tfidf_matrix[livre_id].get(mot, 0.0)
    
    def afficher_tfidf_pour_livre(self, livre_id: int, limite: int = 15):
        """
        Affiche les scores TF-IDF pour un livre spécifique
        
        Args:
            livre_id (int): ID du livre
            limite (int): Nombre maximum de mots à afficher
        """
        if not hasattr(self, 'tfidf_matrix'):
            print("❌ TF-IDF non calculé")
            return
        
        if livre_id not in self.tfidf_matrix:
            print(f"❌ Livre {livre_id} non trouvé")
            return
        
        livre = self.obtenir_livre(livre_id)
        titre = livre.get('titre', 'Inconnu') if livre else 'Inconnu'
        tfidf_dict = self.tfidf_matrix[livre_id]
        
        print(f"\n📖 TF-IDF pour : {titre} (ID: {livre_id})")
        print("-" * 55)
        
        mots_tries = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:limite]
        
        if mots_tries:
            max_score = mots_tries[0][1]
        else:
            max_score = 1
        
        for mot, tfidf in mots_tries:
            barre_longueur = int((tfidf / max_score) * 50) if max_score > 0 else 0
            barre = "█" * barre_longueur
            
            tf = self.obtenir_tf(livre_id, mot)
            idf = self.obtenir_idf(mot)
            print(f"  {mot:15s} : TF-IDF = {tfidf:.6f}  (TF={tf:.4f}, IDF={idf:.4f}) {barre}")
    
    def comparer_tfidf_pour_mot(self, mot: str):
        """
        Compare le score TF-IDF d'un mot dans tous les livres
        
        Args:
            mot (str): Le mot à comparer
        """
        if not hasattr(self, 'tfidf_matrix'):
            print("❌ TF-IDF non calculé")
            return
        
        print(f"\n🔍 Comparaison du mot '{mot}' dans tous les livres :")
        print("-" * 60)
        
        resultats = []
        for livre_id in self.obtenir_tous_les_ids():
            tfidf = self.obtenir_tfidf(livre_id, mot)
            if tfidf > 0:
                livre = self.obtenir_livre(livre_id)
                titre = livre.get('titre', 'Inconnu') if livre else 'Inconnu'
                resultats.append((titre, tfidf, livre_id))
        
        resultats.sort(key=lambda x: x[1], reverse=True)
        
        if not resultats:
            print(f"  Le mot '{mot}' n'apparaît dans aucun livre.")
            return
        
        print(f"  Le mot '{mot}' apparaît dans {len(resultats)} livre(s) :")
        for titre, tfidf, livre_id in resultats:
            barre = "█" * int(tfidf * 100)
            print(f"    [{livre_id}] {titre[:35]:35s} : TF-IDF = {tfidf:.6f} {barre}")
    
    # ================================================================
    # MÉTHODES D'ACCÈS
    # ================================================================
    
    def obtenir_tokens(self, livre_id: int) -> List[str]:
        """Récupère les tokens d'un livre"""
        return self.index.get(livre_id, [])
    
    def obtenir_livre(self, livre_id: int) -> Optional[Dict]:
        """Récupère les informations d'un livre"""
        for livre in self.livres:
            if livre.get('id') == livre_id:
                return livre
        return None
    
    def obtenir_tous_les_livres(self) -> List[Dict]:
        """Récupère tous les livres"""
        return self.livres
    
    def obtenir_tous_les_ids(self) -> List[int]:
        """Récupère tous les IDs"""
        return list(self.index.keys())
    
    def nombre_livres(self) -> int:
        """Retourne le nombre de livres"""
        return len(self.index)
    
    def afficher_index(self, limite: int = 10):
        """Affiche un aperçu de l'index"""
        print("\n" + "=" * 60)
        print("📊 APERÇU DE L'INDEX")
        print("=" * 60)
        
        for i, (livre_id, tokens) in enumerate(self.index.items()):
            if i >= limite:
                print(f"  ... et {len(self.index) - limite} autres livres")
                break
            
            livre = self.obtenir_livre(livre_id)
            titre = livre.get('titre', 'Inconnu') if livre else 'Inconnu'
            genre = livre.get('genre', 'Inconnu') if livre else 'Inconnu'
            
            print(f"\n  [{livre_id}] {titre}")
            print(f"       Genre : {genre}")
            print(f"       Tokens : {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
            print(f"       Nombre total : {len(tokens)} tokens")
    
    # ================================================================
    # STATISTIQUES
    # ================================================================
    
    def statistiques(self) -> Dict:
        """Calcule les statistiques de l'index"""
        if not self.index:
            return {"nb_livres": 0, "nb_tokens_total": 0, "moyenne_tokens_par_livre": 0, "tokens_uniques": 0, "livres_par_genre": {}}
        
        tous_les_tokens = [t for tokens in self.index.values() for t in tokens]
        
        livres_par_genre = {}
        for livre in self.livres:
            genre = livre.get('genre', 'Non spécifié')
            livres_par_genre[genre] = livres_par_genre.get(genre, 0) + 1
        
        return {
            "nb_livres": len(self.index),
            "nb_tokens_total": len(tous_les_tokens),
            "moyenne_tokens_par_livre": len(tous_les_tokens) / len(self.index),
            "tokens_uniques": len(set(tous_les_tokens)),
            "livres_par_genre": livres_par_genre
        }
    
    def afficher_statistiques(self):
        """Affiche les statistiques de l'index"""
        stats = self.statistiques()
        
        print("\n" + "=" * 60)
        print("📊 STATISTIQUES DE L'INDEX")
        print("=" * 60)
        print(f"  📚 Nombre de livres : {stats['nb_livres']}")
        print(f"  🔤 Total des tokens : {stats['nb_tokens_total']}")
        print(f"  📈 Moyenne tokens/livre : {stats['moyenne_tokens_par_livre']:.2f}")
        print(f"  🆕 Tokens uniques : {stats['tokens_uniques']}")
        
        print("\n  📂 Répartition par genre :")
        for genre, count in sorted(stats['livres_par_genre'].items(), key=lambda x: x[1], reverse=True):
            barre = "█" * (count * 2)
            print(f"     {genre:15s} : {count:2d} livre(s) {barre}")
        print("=" * 60)


# ================================================================
# TESTS COMPLETS (ÉTAPES 3.1 À 3.4)
# ================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧪 TEST DU MODULE INDEXEUR (ÉTAPES 3.1, 3.2, 3.3 ET 3.4)")
    print("=" * 70)
    
    # Création de l'indexeur
    indexeur = Indexeur("data/livres.json", "data/stop_words_fr.txt")
    
    # Affichage des statistiques
    indexeur.afficher_statistiques()
    
    # Aperçu de l'index
    indexeur.afficher_index(limite=3)
    
    # Test des scores TF
    print("\n" + "=" * 60)
    print("📊 TEST DES SCORES TF (ÉTAPE 3.2)")
    print("=" * 60)
    
    indexeur.afficher_tf_pour_livre(1, limite=10)
    indexeur.afficher_tf_pour_livre(2, limite=10)
    
    # Test des scores IDF
    print("\n" + "=" * 60)
    print("📊 TEST DES SCORES IDF (ÉTAPE 3.3)")
    print("=" * 60)
    
    mots_a_tester = ["prince", "amour", "justice", "mer", "aventure", "poesie", "classique"]
    indexeur.afficher_idf_pour_mots(mots_a_tester)
    
    # Test des scores TF-IDF
    print("\n" + "=" * 60)
    print("📊 TEST DES SCORES TF-IDF (ÉTAPE 3.4)")
    print("=" * 60)
    
    # Afficher les top TF-IDF pour un livre
    indexeur.afficher_tfidf_pour_livre(1, limite=10)
    
    # Comparer un mot spécifique dans tous les livres
    print("\n" + "=" * 60)
    print("🔍 COMPARAISON TF-IDF POUR UN MOT SPÉCIFIQUE")
    print("=" * 60)
    
    indexeur.comparer_tfidf_pour_mot("amour")
    indexeur.comparer_tfidf_pour_mot("justice")
    indexeur.comparer_tfidf_pour_mot("mer")
    
    # Test de récupération d'un TF-IDF spécifique
    print("\n" + "=" * 60)
    print("🎯 TEST DE RÉCUPÉRATION D'UN TF-IDF SPÉCIFIQUE")
    print("=" * 60)
    
    tfidf_amour_petit_prince = indexeur.obtenir_tfidf(1, "amour")
    print(f"  TF-IDF de 'amour' dans 'Le Petit Prince' : {tfidf_amour_petit_prince:.6f}")
    
    tfidf_justice_miserables = indexeur.obtenir_tfidf(2, "justice")
    print(f"  TF-IDF de 'justice' dans 'Les Misérables' : {tfidf_justice_miserables:.6f}")
    
    tfidf_mer_20000_lieues = indexeur.obtenir_tfidf(3, "mer")
    print(f"  TF-IDF de 'mer' dans 'Vingt Mille Lieues' : {tfidf_mer_20000_lieues:.6f}")
    
    print("\n" + "=" * 70)
    print("✅ ÉTAPES 3.1, 3.2, 3.3 ET 3.4 TERMINÉES AVEC SUCCÈS !")
    print("=" * 70)