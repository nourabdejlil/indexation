"""
Application web Flask pour le moteur de recherche de livres
Étape 5 : Interface web - Routes complètes
"""

from flask import Flask, render_template, request, redirect, url_for, flash
import sys
import os

# Ajout du chemin pour les imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.indexeur import Indexeur
from modules.recherche import MoteurRecherche

# Création de l'application Flask
app = Flask(__name__)
app.secret_key = "votre_cle_secrete_pour_les_sessions"

# Initialisation du moteur de recherche
print("🚀 Démarrage de l'application...")
print("📚 Chargement de l'indexeur...")
indexeur = Indexeur("data/livres.json", "data/stop_words_fr.txt")
print("🔍 Chargement du moteur de recherche...")
moteur = MoteurRecherche(indexeur)
print("✅ Application prête !")


# ================================================================
# ROUTE 1 : Page d'accueil (formulaire de recherche)
# ================================================================

@app.route('/')
def accueil():
    """
    Route / : Page d'accueil avec le formulaire de recherche
    """
    # Récupérer les statistiques pour l'affichage
    stats = indexeur.statistiques()
    return render_template('index.html', stats=stats, indexeur=indexeur)

# ================================================================
# ROUTE 2 : Page de recherche (traitement et résultats)
# ================================================================


@app.route('/recherche', methods=['GET', 'POST'])
def rechercher():
    """
    Route /recherche : Traite la requête et affiche les résultats
    """
    if request.method == 'POST':
        requete = request.form.get('requete', '')
    else:
        requete = request.args.get('q', '')
    
    if not requete or not requete.strip():
        flash('Veuillez saisir une requête de recherche', 'warning')
        return redirect(url_for('accueil'))
    
    # Effectuer la recherche
    resultats_liste = moteur.rechercher(requete, top_n=20)
    
    # Récupérer les tokens de la requête traitée
    requete_tokens = moteur.pretraiter_requete(requete)
    requete_traitee = ' '.join(requete_tokens)
    
    return render_template('resultats.html', 
                         requete_originale=requete,
                         requete_traitee=requete_traitee,
                         resultats=resultats_liste,
                         nb_resultats=len(resultats_liste))


# ================================================================
# ROUTE 3 : Liste de tous les livres (catalogue)
# ================================================================

@app.route('/livres')
def liste_livres():
    """
    Route /livres : Page qui liste tous les livres
    """
    livres = indexeur.obtenir_tous_les_livres()
    
    # Trier les livres par titre
    livres_tries = sorted(livres, key=lambda x: x.get('titre', ''))
    
    return render_template('livres.html', livres=livres_tries)


# ================================================================
# ROUTE 4 : Détail d'un livre (page individuelle)
# ================================================================

@app.route('/livre/<int:livre_id>')
def detail_livre(livre_id):
    """
    Route /livre/<id> : Page de détail d'un livre
    """
    livre = indexeur.obtenir_livre(livre_id)
    
    if not livre:
        flash('Livre non trouvé', 'danger')
        return redirect(url_for('liste_livres'))
    
    # Récupérer les mots clés importants du livre (TF-IDF élevé)
    mots_cles = moteur.get_mots_cles_importants(livre_id, top_n=15)
    
    # Récupérer les statistiques du livre (nombre de tokens)
    tokens = indexeur.obtenir_tokens(livre_id)
    nb_tokens = len(tokens)
    
    return render_template('livre.html', 
                         livre=livre,
                         mots_cles=mots_cles,
                         nb_tokens=nb_tokens)


# ================================================================
# ROUTE 5 : Statistiques de l'index
# ================================================================

@app.route('/stats')
def statistiques():
    """
    Route /stats : Page des statistiques de l'index
    """
    stats = indexeur.statistiques()
    
    # Récupérer les mots les plus fréquents (IDF faible)
    if hasattr(indexeur, 'idf_scores') and indexeur.idf_scores:
        mots_frequents = sorted(indexeur.idf_scores.items(), key=lambda x: x[1])[:20]
    else:
        mots_frequents = []
    
    # Récupérer les mots les plus rares (IDF élevé)
    if hasattr(indexeur, 'idf_scores') and indexeur.idf_scores:
        mots_rares = sorted(indexeur.idf_scores.items(), key=lambda x: x[1], reverse=True)[:20]
    else:
        mots_rares = []
    
    return render_template('stats.html', 
                         stats=stats,
                         mots_frequents=mots_frequents,
                         mots_rares=mots_rares)


# ================================================================
# ROUTE 6 : Recherche avancée (par auteur ou genre)
# ================================================================

@app.route('/recherche/avancee')
def recherche_avancee():
    """
    Route /recherche/avancee : Recherche filtrée par auteur ou genre
    """
    auteur = request.args.get('auteur', '')
    genre = request.args.get('genre', '')
    
    resultats = []
    
    if auteur:
        resultats = moteur.rechercher_par_auteur(auteur, top_n=20)
        type_recherche = f"auteur : {auteur}"
    elif genre:
        resultats = moteur.rechercher_par_genre(genre, top_n=20)
        type_recherche = f"genre : {genre}"
    else:
        flash('Veuillez spécifier un auteur ou un genre', 'warning')
        return redirect(url_for('accueil'))
    
    return render_template('recherche_avancee.html', 
                         resultats=resultats,
                         type_recherche=type_recherche,
                         nb_resultats=len(resultats))


# ================================================================
# LANCEMENT DE L'APPLICATION
# ================================================================

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🌐 Démarrage du serveur web...")
    print("📱 Routes disponibles :")
    print("   - http://127.0.0.1:5000/          (Accueil)")
    print("   - http://127.0.0.1:5000/livres    (Catalogue)")
    print("   - http://127.0.0.1:5000/stats     (Statistiques)")
    print("⏹️  Appuie sur CTRL+C pour arrêter le serveur")
    print("=" * 50 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)