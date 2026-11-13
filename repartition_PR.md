# Répartition des PR — projet MLOps Ride Classifier

13 nov. 2026 · @Chanelle

## Pourquoi on fait ça

Tout le projet MLOps existe déjà dans le draft de Rania (`raniamehria/mlops-ride-classifier-draft`) : 19 commits, du modèle au déploiement Kubernetes. On le ramène dans **notre repo final** sous forme de **18 pull requests relues**, plus des PR réservées pour Advanced Cloud Infra. Le prof voit ainsi chaque étape arriver par une PR ouverte par l'une de nous et relue par une autre, et chacune a une partie claire à défendre à la soutenance.

Les règles :

- **Une PR, un reviewer.** Celui qui ouvre la PR ne la merge pas. Le reviewer la lit, approuve et merge.
- **Merge avec « Create a merge commit ».** Pas « Squash and merge » : le squash réécrit les commits et perd l'historique.
- **Respecter l'ordre.** Chaque PR part de `main` après le merge de la précédente. Dans le désordre, les cherry-picks vont entrer en conflit.
- **Ton travail reste le tien.** Tout ce que tu écris toi-même (PR 16, 17, 18 et Cloud Infra), tu le commites toi-même, sous ton nom.
- **La CI doit être verte avant le merge** à partir de la PR 9 (c'est elle qui ajoute `ci.yml`).

> **Critique : identité de l'auteur**
>
> Tous les commits du draft sont signés par Rania (`raniamehria`). Un simple `git cherry-pick` garde l'auteur d'origine : si on ne fait rien, tous les commits du repo noté apparaîtront au nom de Rania dans `git log`, peu importe qui a ouvert la PR. Le prof regarde `git log`, pas qui a ouvert la PR.
>
> Donc après **chaque** cherry-pick, la personne qui fait la PR remet l'auteur à son nom.

## Setup du repo final — une fois, par Rania, avant la PR 1

1. Créer le repo final sur GitHub **vide** : pas de README, pas de `.gitignore`, pas de licence. Sinon la PR 1 (qui crée `README.md` et `.gitignore`) entre en conflit.
2. Ajouter Luli, Wei-Ling et Chanelle comme collaboratrices.
3. Pousser le draft sur une branche que personne ne merge :

```
git remote add final https://github.com/<org>/<repo-final>.git
git push final main:import/draft
```

4. Créer `main` avec un commit vide, pour que la PR 1 ait une base :

```
git switch --orphan main-init
git commit --allow-empty -m "chore: create repository"
git push final main-init:main
```

5. Réglages à faire dans le repo final (sinon les workflows des PR 11, 12 et 15 échouent) :
   - variable `DOCKERHUB_USERNAME` et secret `DOCKERHUB_TOKEN` (Settings → Secrets and variables → Actions) ;
   - un **runner self-hosted** macOS ARM64 enregistré sur le repo final, sur le Mac qui fera tourner Kubernetes (Docker Desktop) ;
   - variable `RIDE_DATA_SOURCE` = chemin des données sur ce Mac (sinon le workflow de réentraînement prend le chemin du draft) ;
   - protection de `main` : PR obligatoire, 1 approbation, CI verte (à activer après la PR 9).

## Setup — une fois par personne, avant ta première PR

Dans ton clone du repo final, configure ton identité git. Utilise le nom et l'email de ton compte GitHub, pour que les commits soient liés à ton profil :

```
git config user.name "Ton Nom"
git config user.email "ton@email"
```

Vérifie :

```
git config user.name
git config user.email
```

Fais-le **avant** tout cherry-pick : `--reset-author` (voir plus bas) lit ces valeurs.

Puis récupère une fois la branche du draft, pour avoir les commits en local :

```
git fetch origin import/draft
```

## Comment faire une PR

Pour chaque PR, la personne qui l'ouvre (terminal, dans son clone du repo final) :

1. Partir du dernier `main` (exemple : PR 4, Chanelle) :

```
git fetch origin
git switch -c pr4/chanelle/mlflow-minio origin/main
```

2. Copier les commits dans l'ordre du tableau. Après chaque cherry-pick, remettre l'auteur à soi :

```
git cherry-pick e5faedd
git commit --amend --reset-author --no-edit
```

> `--reset-author` remplace le nom et l'email par ta config git locale. À faire pour chaque commit de **tes** PR uniquement, jamais sur les commits de la PR de quelqu'un d'autre.

3. Vérifier que tu es bien l'auteur avant de pousser :

```
git log --format='%h %an <%ae> %s' origin/main..HEAD
```

Chaque ligne doit montrer ton nom et ton email. Si une ligne montre encore Rania, reviens à l'étape 2 pour ce commit.

4. Pousser la branche :

```
git push -u origin pr4/chanelle/mlflow-minio
```

5. Sur GitHub : **Compare & pull request**, base `main`. Titre : le numéro et ce que la PR ajoute. Ajouter le reviewer du tableau.

**Le reviewer :** ouvrir *Files changed*, lire, laisser un commentaire si quelque chose n'est pas clair, puis *Review changes → Approve* et *Merge pull request → Create a merge commit*. Prévenir la personne suivante sur le groupe.

**Si un cherry-pick s'arrête sur un conflit :** ne pas le corriger à la main. Lancer `git cherry-pick --abort` et envoyer le numéro de la PR à Rania.

## Les PR, dans l'ordre

Une à la fois : on ouvre la suivante seulement quand la précédente est mergée. Mettez le statut à jour au fur et à mesure.

### Socle obligatoire (PR 1 à 12)

| PR | Ouvre | Relit et merge | Commits à cherry-pick | Dépend de | Ce que ça ajoute (exigence des consignes) | Statut |
|---|---|---|---|---|---|---|
| 1 | Rania | Chanelle | `a59f291` | — | Structure du projet avec uv (`src/`, `tests/`, `docker/`, `k8s/`…) | À faire |
| 2 | Luli | Wei-Ling | `3a25522` `7ccf2cd` | 1 | Exploration des données, preprocessing (nettoyage, fenêtres de 60 s, split par trajet, normalisation) | À faire |
| 3 | Luli | Chanelle | `676b459` `9db5aae` | 2 | Évaluation par trajet + baselines, **CNN 1D 5 classes** avec poids de classes (modèle deep learning) | À faire |
| 4 | Chanelle | Wei-Ling | `e5faedd` | 1 | Stack dev Docker Compose : **MLflow + PostgreSQL + MinIO (S3)** | À faire |
| 5 | Chanelle | Luli | `ac82476` | 3, 4 | **Tracking MLflow + Model Registry**, alias `champion`, modèle stocké sur MinIO | À faire |
| 6 | Chanelle | Rania | `4db6f88` | 5 | **API FastAPI** qui sert le champion depuis MLflow (`/predict`, `/health`, `/model`) | À faire |
| 7 | Chanelle | Luli | `0fba9c4` | 6 | **Web app Streamlit** pour tester l'API | À faire |
| 8 | Wei-Ling | Chanelle | `d1ffe7a` | 6, 7 | **Dockerfiles API + web app**, toute la stack de dev dans Compose | À faire |
| 9 | Rania | Luli | `9c0c143` | 8 | **Tests unitaires, d'intégration et e2e**, ruff, workflow CI. Activer la protection de `main` après le merge | À faire |
| 10 | Wei-Ling | Rania | `9bc531b` | 8 | **Prod sur Kubernetes local** (manifests kustomize, namespace, 2 replicas API) | À faire |
| 11 | Wei-Ling | Rania | `82eeed8` | 8 | **Images multi-arch publiées sur Docker Hub**. Vérifier que le workflow passe dans l'onglet Actions | À faire |
| 12 | Wei-Ling | Chanelle | `90495dd` | 10, 11 | **Déploiement GitHub Actions sur runner self-hosted**, smoke tests, rollback auto. Vérifier un déploiement réel | À faire |

### Bonus (PR 13 à 16)

| PR | Ouvre | Relit et merge | Commits à cherry-pick | Dépend de | Ce que ça ajoute | Statut |
|---|---|---|---|---|---|---|
| 13 | Luli | Wei-Ling | `102a14f` | 5, 8 | Pipeline d'entraînement Airflow (check data → preprocess → train → reload API) | À faire |
| 14 | Rania | Wei-Ling | `33b633a` | 6, 8 | **Monitoring** : métriques Prometheus, alertes, dashboard Grafana | À faire |
| 15 | Chanelle | Rania | `0b585d5` | 5, 12 | **Pipeline de réentraînement** GitHub Actions, redéploie seulement si le modèle est meilleur | À faire |
| 16 | Luli | Rania | Son propre commit, **à écrire** | 3, 5 | **Feature store** (n'existe pas dans le draft) : par ex. Feast avec un offline store parquet, features par trajet, utilisées à l'entraînement, tests unitaires | À faire |

### Documentation et clôture (PR 17 et 18)

| PR | Ouvre | Relit et merge | Commits à cherry-pick | Dépend de | Ce que ça ajoute | Statut |
|---|---|---|---|---|---|---|
| 17 | Rania | Luli | `c5f4be4` `de6c6c7` + son propre commit | 1 à 16 | **README** (architecture, choix techniques, résultats, captures). Son commit ajoute : nom du repo final, section feature store, liste du groupe | À faire |
| 18 | Chacune | Une autre membre | Ton propre commit | 17 | Ta ligne dans la section « Contributions » du README (voir plus bas) | À faire |

> **Note sur les PR 16, 17 et 18 :** dans ces PR tu écris ton propre commit, tu es donc déjà l'auteur. `--reset-author` n'est pas nécessaire. L'étape de reset s'applique à chaque commit **cherry-pické** : PR 1 à 15 et les deux premiers commits de la PR 17.

Noms de branche : `pr<N>/<ton-prénom>/<sujet>`, par exemple `pr4/chanelle/mlflow-minio`.

### Vue des dépendances

Les chaînes à connaître (A → B : B a besoin de A) :

- **Modèle :** 1 → 2 → 3 → 5
- **Stockage et registry :** 1 → 4 → 5
- **Service :** 5 → 6 (API) → 7 (web app) → 8 (Docker)
- **Qualité :** 8 → 9 (tests et CI)
- **Prod :** 8 → 10 (Kubernetes) et 8 → 11 (Docker Hub), puis 10 + 11 → 12 (déploiement)
- **Bonus :** 5 + 8 → 13 (Airflow) · 6 + 8 → 14 (monitoring) · 12 → 15 (réentraînement) · 3 + 5 → 16 (feature store)
- **Clôture :** 1 à 16 → 17 (README) → 18 (contributions) → Cloud Infra

Même quand deux PR ne dépendent pas l'une de l'autre (par ex. 9 et 10), on garde l'ordre du tableau : les cherry-picks suivent l'ordre des commits du draft.

## Rôles et charge par membre

| Membre | Rôle (ce qu'elle défend à l'oral) | Ouvre | Relit | Total |
|---|---|---|---|---|
| Rania | **Architecture, qualité et observabilité** : structure du projet, tests et CI, monitoring Prometheus/Grafana, README. Elle a écrit le draft : c'est elle qui débloque les conflits | 1, 9, 14, 17 + setup du repo | 6, 10, 11, 15, 16 | 4 + 5 |
| Luli | **Données et modèle** : exploration, preprocessing, CNN et métriques, pipeline Airflow, feature store | 2, 3, 13, 16 (nouveau code) | 5, 7, 9, 17 | 4 + 4 |
| Wei-Ling | **Plateforme et déploiement** : Dockerfiles, images Docker Hub, Kubernetes, déploiement GitHub Actions sur le runner self-hosted | 8, 10, 11, 12 | 2, 4, 13, 14 | 4 + 4 |
| Chanelle | **Cycle de vie du modèle** : MinIO, tracking et registry MLflow, alias champion, API et web app qui servent le champion, réentraînement | 4, 5, 6, 7, 15 | 1, 3, 8, 12 | 5 + 4 |

Pourquoi ces rôles :

- **Rania** garde ce qu'elle maîtrise le mieux dans son draft : la vue d'ensemble, la qualité et la doc. Le README et le monitoring demandent de connaître toute la stack.
- **Luli** prend toute la partie data science, du fichier brut aux features. Le feature store est le seul gros morceau à écrire, il est donc avec ses autres PR, plus légères.
- **Wei-Ling** prend toute la chaîne conteneurs → Kubernetes → CD. Ce sont les PR les plus longues à vérifier (il faut un vrai déploiement sur le runner).
- **Chanelle** suit le modèle de son enregistrement à sa mise en service : MLflow, registry, API, web app, puis réentraînement. Elle ouvre une PR de plus, mais la PR 7 (web app) est petite.

Chacune relit au moins une partie qu'elle n'a pas ouverte : le prof interroge chacune sur une partie qu'elle n'a pas faite.

## Contributions (PR 18)

Pour la soutenance, le README dit qui a fait quoi. Après le merge de la PR 17, chacune ajoute sa ligne dans la section « Contributions » du README, puis commite et ouvre sa propre PR :

```
- **<Ton prénom> :** <ton rôle> — PR <numéros> (défend ces parties à la soutenance)
```

Lis attentivement le code de tes PR, et au moins une PR que tu n'as pas ouverte. Le rôle de chacune est dans la section précédente.

## Advanced Cloud Infra (PR 19 et suivantes)

Les consignes d'Advanced Cloud Infra **ne sont pas encore sorties**. On réserve les PR 19 et suivantes pour ce cours, et on ne prévoit rien de plus tant qu'on ne les a pas.

| PR | Ouvre | Relit et merge | Contenu | Dépend de | Statut |
|---|---|---|---|---|---|
| 19+ | À répartir | À répartir | À définir dès la sortie des consignes | Au moins 12 (prod en place) | En attente |

Quand les consignes arrivent : on liste les exigences, on fait une PR par exigence, et on commence par Luli, Wei-Ling et Rania (une PR de moins que Chanelle dans le MLOps) avant de reboucler sur tout le monde.

## Points à vérifier avant de commencer

- **Image ou CSV ?** Les consignes parlent d'une API qui prend une **image**. Le draft prend un **CSV** de trajet, parce que les données sont des séries de capteurs et pas des images (justifié dans le README). À confirmer avec les profs pour ne pas perdre de points.
- **Docker Hub :** les images et `k8s/kustomization.yaml` pointent sur le compte `raniamehria`. Soit on garde ce compte (il faut alors son token dans le repo final), soit on change le nom dans la PR 10 et la PR 11.
- **Auteur du projet :** `pyproject.toml` contient le nom et l'email de Rania. À mettre à jour dans la PR 17 si on veut le groupe complet.

## Deadline et avant la soutenance

- **PR 1 à 12 mergées le 18 nov. au soir** (socle obligatoire).
- **PR 13 à 16 mergées le 21 nov. au soir** (bonus, le feature store est à écrire).
- **PR 17 et 18 mergées le 23 nov. au soir.** Cloud Infra selon la date de ses consignes.
- **Liste du groupe envoyée aux profs pour le 20 nov.**
- **QCM le 24 nov. à 17 h.**
- **Rendu le 25 nov. à 20 h** : mail `[ALBERT][MLOPS]` aux deux profs avec le lien du repo et la liste du groupe.
- **Soutenance le 26 nov.** : 10 min de présentation + 10 min de questions.

Checklist :

- [ ] Les 18 PR mergées, dans l'ordre, chacune relue par une autre membre
- [ ] Chaque commit du repo final au nom de celle qui a ouvert la PR (`git log`)
- [ ] La CI, la publication Docker Hub et le déploiement sur Kubernetes passent dans l'onglet Actions du repo final
- [ ] Chacune a ajouté sa ligne de contributions (PR 18)
- [ ] Chacune a relu ses PR et au moins une qu'elle n'a pas ouverte
- [ ] Démo testée sur le Mac du runner : web app → API → prédiction, Grafana, MLflow
- [ ] Une répétition chronométrée de la présentation de 10 min, avec les passages de parole
