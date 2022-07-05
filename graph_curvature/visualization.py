import networkx as nx
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from graph_curvature.curvature import GraphCurvature


def make_curv_table(G_initial, tcat, tcat_anno, curvatures_csv='scalar_curvatures_C800_med_conf.csv'):
    extra_nodes = set(G_initial.nodes) - set(tcat['gene'].tolist())
    G_initial.remove_nodes_from(extra_nodes)
    G = [G_initial.subgraph(c).copy() for c in nx.connected_components(G_initial) if
         c == max(nx.connected_components(G_initial), key=len)][0]
    common_nodes = list(set(tcat['gene'].tolist()).intersection(G.nodes))
    tcat = tcat[tcat['gene'].apply(lambda x: x in common_nodes)]

    curv = pd.read_csv(curvatures_csv, names=['gene', 'curvature'], skiprows=1)
    curv = curv[curv['gene'].apply(lambda x: x in common_nodes)]
    curv['curvature'] = curv['curvature'] * 100

    orc = GraphCurvature.from_save(G, curv)

    tc_curv_df, tc_nodal_curvs = orc.curvature_per_pat(tcat)
    tc_curv_table = tc_curv_df.merge(tcat_anno, left_on='subject', right_on='SampleID')
    tc_curv_table['# antigen exposures'] = tc_curv_table['# antigen exposures'].apply(
        lambda x: 0 if x == 'NA (pre-stimulation sample)' else int(x))
    return tc_curv_table, tc_nodal_curvs


def plot_curvature_per_donor(tcat_curv_table):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.set(font_scale=1)
    sfig = sns.lineplot(ax=ax, data=tcat_curv_table, x='# antigen exposures', y='curvature', hue='Donor',
                        palette='tab10')
    sfig.set_xlabel('# Antigen Exposures', fontsize=20)
    sfig.set_ylabel('Curvature (unitless)', fontsize=20)
    return


def plot_curvature_for_gene(tc_nodal_curvs, tcat_donors, gene):
    tc = tc_nodal_curvs[tcat_donors.loc[3]]
    tc = tc.loc[tc.diff(axis=1).mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.set(font_scale=1.5)
    plt.xticks(rotation=75)
    sns.lineplot(ax=ax, data=tc.loc[gene])
    return

