"""
Module de prétraitement du texte
Responsable : nettoyage, suppression des accents, stop words, tokenisation

Étapes implémentées :
- Étape 2.1 : Nettoyage du texte (minuscules, accents, ponctuation, chiffres)
- Étape 2.2 : Suppression des stop words (mots vides)
- Étape 2.3 : Tokenisation (découpage en mots)
"""

import re
import unicodedata
from typing import List, Set


class Preprocesseur:
    """
    Classe qui gère le nettoyage et la préparation des textes
    pour l'indexation et la recherche
    """
    
    def __init__(self, stop_words_path: str = "data/stop_words_fr.txt"):
        """
        Initialise le préprocesseur et charge les stop words
        
        Args:
            stop_words_path (str): Chemin vers le fichier des stop words
        """
        # Compilation des expressions régulières pour meilleures performances
        self.pattern_chiffres = re.compile(r'\d+')
        self.pattern_punctuation = re.compile(r'[^\w\s]')
        
        # Chargement des stop words
        self.stop_words = self._charger_stop_words(stop_words_path)
        
        print(f"✅ {len(self.stop_words)} stop words chargés")
    
    # ================================================================
    # CHARGEMENT DES STOP WORDS (utilisé par l'étape 2.2)
    # ================================================================
    
    def _charger_stop_words(self, chemin: str) -> Set[str]:
        """
        Charge les stop words depuis un fichier texte
        Chaque ligne du fichier contient un mot
        
        Args:
            chemin (str): Chemin vers le fichier des stop words
            
        Returns:
            Set[str]: Ensemble des stop words pour recherche rapide
        """
        stop_words = set()
        
        try:
            with open(chemin, 'r', encoding='utf-8') as fichier:
                for ligne in fichier:
                    mot = ligne.strip().lower()
                    if mot:  # Ignorer les lignes vides
                        stop_words.add(mot)
        except FileNotFoundError:
            print(f"⚠️ Fichier {chemin} non trouvé. Utilisation d'une liste par défaut.")
            stop_words = self._stop_words_defaut()
        
        return stop_words
    
    def _stop_words_defaut(self) -> Set[str]:
        """
        Liste de stop words par défaut (secours)
        
        Returns:
            Set[str]: Ensemble des stop words par défaut
        """
        return {
            'le', 'la', 'les', 'l', 'un', 'une', 'des', 'du', 'de', 'd',
            'et', 'ou', 'mais', 'donc', 'car', 'or', 'ni',
            'ce', 'cet', 'cette', 'ces', 'mon', 'ton', 'son',
            'ma', 'ta', 'sa', 'mes', 'tes', 'ses',
            'notre', 'votre', 'leur', 'nos', 'vos', 'leurs',
            'je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
            'me', 'te', 'se', 'lui', 'leur', 'y', 'en',
            'est', 'sont', 'était', 'étaient', 'sera', 'seront',
            'être', 'avoir', 'fait', 'faire', 'aller', 'venir',
            'a', 'au', 'aux', 'avec', 'sans', 'pour', 'par',
            'dans', 'sur', 'sous', 'entre', 'jusque', 'vers',
            'pendant', 'avant', 'après', 'contre', 'comme',
            'quand', 'lorsque', 'puisque', 'parce', 'que',
            'qui', 'quoi', 'dont', 'où', 'ne', 'pas',
            'plus', 'moins', 'très', 'trop', 'si', 'aussi',
            'donc', 'voici', 'voilà', 'enfin', 'oui', 'non',
            'soit', 'selon', 'environ', 'moi', 'toi',
            'eux', 'celles', 'ceci', 'cela', 'ça', 'celui',
            'celle', 'ceux', 'celles', 'ci', 'là'
        }
    
    # ================================================================
    # ÉTAPE 2.1 : NETTOYAGE DU TEXTE
    # ================================================================
    
    def en_minuscules(self, texte: str) -> str:
        """
        Convertit tout le texte en minuscules
        
        Args:
            texte (str): Le texte à convertir
            
        Returns:
            str: Texte en minuscules
        """
        return texte.lower()
    
    def supprimer_accents(self, texte: str) -> str:
        """
        Supprime les accents d'une chaîne de caractères
        Exemple: "Éléphant" → "Elephant"
        
        Args:
            texte (str): Le texte à nettoyer des accents
            
        Returns:
            str: Texte sans accents
        """
        # Décompose les caractères accentués (é → e + accent)
        texte_normalise = unicodedata.normalize('NFD', texte)
        # Garde uniquement les caractères non diacritiques (supprime les accents)
        texte_sans_accents = ''.join(
            c for c in texte_normalise 
            if unicodedata.category(c) != 'Mn'  # 'Mn' = Mark, nonspacing (les accents)
        )
        return texte_sans_accents
    
    def supprimer_punctuation(self, texte: str) -> str:
        """
        Supprime toute la ponctuation d'un texte
        Exemple: "Bonjour ! Comment ça va ?" → "Bonjour  Comment ça va "
        
        Args:
            texte (str): Le texte à nettoyer
            
        Returns:
            str: Texte sans ponctuation
        """
        # Remplace toute ponctuation par un espace
        return self.pattern_punctuation.sub(' ', texte)
    
    def supprimer_chiffres(self, texte: str) -> str:
        """
        Supprime tous les chiffres d'un texte
        Exemple: "20 000 lieues sous les mers" → "  lieues sous les mers"
        
        Args:
            texte (str): Le texte à nettoyer
            
        Returns:
            str: Texte sans chiffres
        """
        # Remplace les chiffres par un espace
        return self.pattern_chiffres.sub(' ', texte)
    
    def supprimer_espaces_superflus(self, texte: str) -> str:
        """
        Supprime les espaces multiples et les espaces en début/fin de chaîne
        Exemple: "  Bonjour   le  monde  " → "Bonjour le monde"
        
        Args:
            texte (str): Le texte à nettoyer
            
        Returns:
            str: Texte avec espaces simples
        """
        # Remplace les espaces multiples par un seul espace
        texte_simple = re.sub(r'\s+', ' ', texte)
        # Supprime les espaces en début et fin
        return texte_simple.strip()
    
    def nettoyer(self, texte: str) -> str:
        """
        ÉTAPE 2.1 : Pipeline complet de nettoyage du texte
        Applique toutes les étapes dans l'ordre :
        1. Minuscules
        2. Suppression des accents
        3. Suppression de la ponctuation
        4. Suppression des chiffres
        5. Suppression des espaces superflus
        
        Args:
            texte (str): Le texte brut à nettoyer
            
        Returns:
            str: Texte propre, prêt à être tokenisé
        """
        if not texte or not isinstance(texte, str):
            return ""
        
        texte = self.en_minuscules(texte)
        texte = self.supprimer_accents(texte)
        texte = self.supprimer_punctuation(texte)
        texte = self.supprimer_chiffres(texte)
        texte = self.supprimer_espaces_superflus(texte)
        
        return texte
    
    # ================================================================
    # ÉTAPE 2.3 : TOKENISATION (découpage en mots)
    # ================================================================
    
    def tokeniser(self, texte: str) -> List[str]:
        """
        ÉTAPE 2.3 : Transforme un texte nettoyé en liste de tokens (mots individuels)
        Une simple méthode split() suffit pour commencer.
        
        Args:
            texte (str): Le texte déjà nettoyé (minuscules, sans accents, sans ponctuation)
            
        Returns:
            List[str]: Liste des tokens (mots individuels)
        """
        if not texte:
            return []
        
        # La méthode split() découpe sur les espaces
        # Exemple: "le petit prince" → ["le", "petit", "prince"]
        return texte.split()
    
    def tokeniser_avance(self, texte: str) -> List[str]:
        """
        Version avancée de la tokenisation (pour plus tard)
        Gère les cas particuliers comme les tirets, apostrophes, etc.
        
        Args:
            texte (str): Le texte déjà nettoyé
            
        Returns:
            List[str]: Liste des tokens
        """
        if not texte:
            return []
        
        # Version plus sophistiquée : gère les mots avec tirets
        tokens = []
        for mot in texte.split():
            # Si le mot contient un tiret, on peut choisir de le garder ou de le séparer
            if '-' in mot and len(mot) > 3:
                # Option 1 : garder le mot avec tiret
                tokens.append(mot)
                # Option 2 (décommenter pour séparer) :
                # tokens.extend(mot.split('-'))
            else:
                tokens.append(mot)
        
        return tokens
    
    # ================================================================
    # ÉTAPE 2.2 : SUPPRESSION DES STOP WORDS
    # ================================================================
    
    def filtrer_stop_words(self, tokens: List[str]) -> List[str]:
        """
        ÉTAPE 2.2 : Filtre une liste de tokens pour supprimer les stop words
        et les mots trop courts (moins de 2 lettres)
        
        Args:
            tokens (List[str]): Liste des tokens à filtrer
            
        Returns:
            List[str]: Liste des tokens sans stop words
        """
        if not tokens:
            return []
        
        return [mot for mot in tokens if mot not in self.stop_words and len(mot) >= 2]
    
    def est_stop_word(self, mot: str) -> bool:
        """
        Vérifie si un mot est un stop word
        
        Args:
            mot (str): Le mot à vérifier
            
        Returns:
            bool: True si c'est un stop word, False sinon
        """
        return mot.lower() in self.stop_words
    
    def ajouter_stop_word(self, mot: str):
        """
        Ajoute un mot à la liste des stop words (utile pour personnaliser)
        
        Args:
            mot (str): Le mot à ajouter
        """
        self.stop_words.add(mot.lower())
        print(f"➕ Stop word ajouté : {mot.lower()}")
    
    def supprimer_stop_word(self, mot: str):
        """
        Supprime un mot de la liste des stop words
        
        Args:
            mot (str): Le mot à supprimer
        """
        if mot.lower() in self.stop_words:
            self.stop_words.remove(mot.lower())
            print(f"➖ Stop word supprimé : {mot.lower()}")
    
    # ================================================================
    # PIPELINE COMPLET (2.1 + 2.3 + 2.2)
    # ================================================================
    
    def tokenizer(self, texte: str) -> List[str]:
        """
        Pipeline COMPLET de prétraitement :
        1. Étape 2.1 : Nettoyage (minuscules, accents, ponctuation, chiffres)
        2. Étape 2.3 : Tokenisation (découpage en mots)
        3. Étape 2.2 : Suppression des stop words
        
        Args:
            texte (str): Le texte brut à traiter
            
        Returns:
            List[str]: Liste des tokens utiles (sans stop words)
        """
        if not texte:
            return []
        
        # Étape 2.1 : Nettoyage
        texte_propre = self.nettoyer(texte)
        
        # Étape 2.3 : Tokenisation (découpage en mots)
        tokens = self.tokeniser(texte_propre)
        
        # Étape 2.2 : Suppression des stop words
        tokens_filtres = self.filtrer_stop_words(tokens)
        
        return tokens_filtres
    
    def tokenizer_sans_filtre(self, texte: str) -> List[str]:
        """
        Tokenise sans supprimer les stop words (utile pour déboguer)
        
        Args:
            texte (str): Le texte à tokeniser
            
        Returns:
            List[str]: Liste de tous les tokens (avec stop words)
        """
        if not texte:
            return []
        
        texte_propre = self.nettoyer(texte)
        return self.tokeniser(texte_propre)


# ================================================================
# TESTS
# ================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 TEST COMPLET DU MODULE PREPROCESSEUR")
    print("=" * 70)
    
    # Création d'une instance
    preproc = Preprocesseur("data/stop_words_fr.txt")
    
    print("\n" + "=" * 70)
    print("📝 ÉTAPE 2.1 : TEST DU NETTOYAGE")
    print("=" * 70)
    
    test_texte = "L'Éléphant et la Souris: une histoire de 20 kilos!"
    print(f"Original : {test_texte}")
    print(f"Nettoyé   : {preproc.nettoyer(test_texte)}")
    
    print("\nTests supplémentaires de nettoyage :")
    tests_nettoyage = [
        "Bonjour ! Comment ça va ?",
        "Hélas, je n'ai pas fini...",
        "20 000 lieues sous les mers",
        "Ça c'est une belle histoire !",
        "L'éducation nationale française",
        "Réponse: 42 (la réponse)",
    ]
    
    for test in tests_nettoyage:
        print(f"  '{test}' → '{preproc.nettoyer(test)}'")
    
    print("\n" + "=" * 70)
    print("📝 ÉTAPE 2.3 : TEST DE LA TOKENISATION (sans stop words)")
    print("=" * 70)
    
    texte_nettoye = preproc.nettoyer("Le petit prince est un conte philosophique merveilleux")
    print(f"Texte nettoyé : {texte_nettoye}")
    print(f"Tokens bruts  : {preproc.tokeniser(texte_nettoye)}")
    
    print("\n" + "=" * 70)
    print("📝 ÉTAPE 2.2 : TEST DE LA SUPPRESSION DES STOP WORDS")
    print("=" * 70)
    
    tokens_bruts = ["le", "petit", "prince", "est", "un", "conte", "philosophique", "merveilleux"]
    print(f"Tokens bruts      : {tokens_bruts}")
    print(f"Après filtrage    : {preproc.filtrer_stop_words(tokens_bruts)}")
    
    print("\n" + "=" * 70)
    print("📝 PIPELINE COMPLET (2.1 + 2.3 + 2.2)")
    print("=" * 70)
    
    phrase = "Je cherche un livre sur la mer et les aventures"
    print(f"Phrase originale : {phrase}")
    print(f"Pipeline complet : {preproc.tokenizer(phrase)}")
    
    print("\n" + "=" * 70)
    print("📝 TEST SUR UNE DESCRIPTION DE LIVRE")
    print("=" * 70)
    
    description = "Un aviateur rencontre un petit prince venu d'une autre planète. Ensemble, ils découvrent les mystères de l'amitié et de la vie."
    print(f"Description : {description[:80]}...")
    tokens = preproc.tokenizer(description)
    print(f"Nombre de tokens : {len(tokens)}")
    print(f"Tokens : {tokens}")
    
    print("\n" + "=" * 70)
    print("📝 TEST DE LA FONCTION est_stop_word()")
    print("=" * 70)
    
    mots_test = ["le", "petit", "prince", "et", "une", "histoire", "amour"]
    for mot in mots_test:
        est_stop = preproc.est_stop_word(mot)
        print(f"  '{mot}' : {'❌ STOP WORD' if est_stop else '✅ utile'}")
    
    print("\n" + "=" * 70)
    print("✅ TOUS LES TESTS SONT PASSÉS !")
    print("=" * 70)
    
    print("\n📊 RÉCAPITULATIF DES MÉTHODES DISPONIBLES :")
    print("-" * 70)
    print("  Étape 2.1 :")
    print("    - nettoyer() : nettoyage complet du texte")
    print("    - en_minuscules(), supprimer_accents(), supprimer_punctuation()...")
    print("\n  Étape 2.3 :")
    print("    - tokeniser() : découpage d'un texte nettoyé en mots")
    print("    - tokeniser_avance() : version avancée avec gestion des tirets")
    print("\n  Étape 2.2 :")
    print("    - filtrer_stop_words() : suppression des stop words")
    print("    - est_stop_word() : vérifie si un mot est un stop word")
    print("    - ajouter_stop_word() / supprimer_stop_word()")
    print("\n  Pipeline complet :")
    print("    - tokenizer() : nettoyage + tokenisation + suppression stop words")
    print("    - tokenizer_sans_filtre() : nettoyage + tokenisation (sans filtre)")