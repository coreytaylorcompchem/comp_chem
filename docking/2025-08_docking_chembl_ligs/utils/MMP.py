import io
import base64

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from operator import itemgetter
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def remove_map_nums(mol):
    """
    Remove atom map numbers from a molecule
    """
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)

def sort_fragments(mol):
    """
    Transform a molecule with multiple fragments into a list of molecules that is sorted by number of atoms
    from largest to smallest
    """
    frag_list = list(Chem.GetMolFrags(mol, asMols=True))
    [remove_map_nums(x) for x in frag_list]
    frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in frag_list]
    frag_num_atoms_list.sort(key=itemgetter(0), reverse=True)
    return [x[1] for x in frag_num_atoms_list]

def rxn_to_base64_image(rxn):
    """
    Convert RDKit reaction to an image
    """
    drawer = rdMolDraw2D.MolDraw2DCairo(300, 150)
    drawer.DrawReaction(rxn)
    drawer.FinishDrawing()
    bio = io.BytesIO()
    text = drawer.GetDrawingText()
    im_text64 = base64.b64encode(text).decode('utf8')
    img_str = f"<img src='data:image/png;base64, {im_text64}'/>"
    return img_str

def strippplot_base64_image_pic50(dist, dist2):
    """
    Plot distribution as stripplot and save the resulting image as a base64 image. 
    """
    sns.set(rc={'figure.figsize': (3, 1)})
    sns.set_style("ticks")
    sns.despine()

    df = pd.DataFrame(
        {'data': dist,
        'recent': dist2,
        })
    df = df.sort_values(by=['recent'], ascending=False)

    palette = sns.color_palette(['rebeccapurple', 'limegreen'])

    ax = sns.stripplot(data = df, x=df.data, hue=df.recent, palette=palette, legend=False, alpha = 0.9)
    ax.axvline(0,ls="--",c="red")
    ax.set_xlim(-4,4)
    ax.set_xlabel('ΔpIC50')
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches="tight")
    plt.close()
    s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return '<img align="left" src="data:image/png;base64,%s">' % s

def strippplot_base64_image_high_affinity(dist):
    """
    Plot distribution as stripplot and save the resulting image as a base64 image. 
    """
    sns.set(rc={'figure.figsize': (3, 1)})
    sns.set_style("ticks")
    sns.despine()
    palette = sns.color_palette("rocket", n_colors = 5)

    ax = sns.stripplot(x=dist, palette=palette)
    ax.axvline(0,ls="--",c="red")
    ax.set_xlim(-2,2)
    ax.set_xlabel('ΔpIC50')
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches="tight")
    plt.close()
    s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return '<img align="left" src="data:image/png;base64,%s">' % s

def strippplot_base64_image_LogD(dist, dist2):
    """
    Plot distribution as stripplot and save the resulting image as a base64 image. 
    """
    sns.set(rc={'figure.figsize': (3, 1)})
    sns.set_style("ticks")
    sns.despine()
    df = pd.DataFrame(
        {'data': dist,
        'recent': dist2,
        })
    df = df.sort_values(by=['recent'], ascending=False)

    palette = sns.color_palette(['rebeccapurple', 'limegreen'])

    ax = sns.stripplot(data = df, x=df.data, hue=df.recent, palette=palette, legend=False, alpha = 0.9)
    ax.axvline(0,ls="--",c="red")
    ax.set_xlim(-3,3)
    ax.set_xlabel('ΔLogD')
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches="tight")
    plt.close()
    s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return '<img align="left" src="data:image/png;base64,%s">' % s

def strippplot_base64_image_MetStab(dist, dist2):
    """
    Plot distribution as stripplot and save the resulting image as a base64 image. 
    """
    sns.set(rc={'figure.figsize': (3, 1)})
    sns.set_style("ticks")
    sns.despine()
    df = pd.DataFrame(
        {'data': dist,
        'recent': dist2,
        })
    df = df.sort_values(by=['recent'], ascending=False)

    palette = sns.color_palette(['rebeccapurple', 'limegreen'])

    ax = sns.stripplot(data = df, x=df.data, hue=df.recent, palette=palette, legend=False, alpha = 0.9)
    ax.axvline(0,ls="--",c="red")
    ax.set_xlim(-1000,1000)
    ax.set_xlabel('ΔCl(int) (µL/min/mg)')
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches="tight")
    plt.close()
    s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return '<img align="left" src="data:image/png;base64,%s">' % s

def strippplot_base64_image_MAP4K3_HPK1(dist, dist2):
    """
    Plot distribution as stripplot and save the resulting image as a base64 image. 
    """
    sns.set(rc={'figure.figsize': (3, 1)})
    sns.set_style("ticks")
    sns.despine()
    palette = sns.color_palette("rocket", n_colors = 5)

    ax = sns.stripplot(x=dist, palette=palette)
    ax.axvline(0,ls="--",c="red")
    ax.set_xlim(-300,300)
    ax.set_xlabel('ΔMAP4K3_HPK1 ratio')
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches="tight")
    plt.close()
    s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return '<img align="left" src="data:image/png;base64,%s">' % s

def strippplot_base64_image_selectivity_index(dist, dist2):
    """
    Plot distribution as stripplot and save the resulting image as a base64 image. 
    """
    sns.set(rc={'figure.figsize': (3, 1)})
    sns.set_style("ticks")
    sns.despine()

    df = pd.DataFrame(
        {'data': dist,
        'recent': dist2,
        })
    df = df.sort_values(by=['recent'], ascending=False)

    palette = sns.color_palette(['rebeccapurple', 'limegreen'])

    ax = sns.stripplot(data = df, x=df.data, hue=df.recent, palette=palette, legend=False, alpha = 0.9)
    ax.axvline(0,ls="--",c="red")
    ax.set_xlim(-300,300)
    ax.set_xlabel('Δselectivity_index')
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches="tight")
    plt.close()
    s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return '<img align="left" src="data:image/png;base64,%s">' % s

def strippplot_base64_image_pslp76(dist, dist2):
    """
    Plot distribution as stripplot and save the resulting image as a base64 image. 
    """
    sns.set(rc={'figure.figsize': (3, 1)})
    sns.set_style("ticks")
    sns.despine()

    df = pd.DataFrame(
        {'data': dist,
        'recent': dist2,
        })
    df = df.sort_values(by=['recent'], ascending=False)

    palette = sns.color_palette(['rebeccapurple', 'limegreen'])

    ax = sns.stripplot(data = df, x=df.data, hue=df.recent, palette=palette, legend=False, alpha = 0.9)
    ax.axvline(0,ls="--",c="red")
    ax.set_xlim(-3,3)
    ax.set_xlabel('ΔpSLP76 pIC50')
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches="tight")
    plt.close()
    s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return '<img align="left" src="data:image/png;base64,%s">' % s

def strippplot_base64_image_il2(dist, dist2):
    """
    Plot distribution as stripplot and save the resulting image as a base64 image. 
    """
    sns.set(rc={'figure.figsize': (3, 1)})
    sns.set_style("ticks")
    sns.despine()

    df = pd.DataFrame(
        {'data': dist,
        'recent': dist2,
        })
    df = df.sort_values(by=['recent'], ascending=False)

    palette = sns.color_palette(['rebeccapurple', 'limegreen'])

    ax = sns.stripplot(data = df, x=df.data, hue=df.recent, palette=palette, legend=False, alpha = 0.9)
    ax.axvline(0,ls="--",c="red")
    ax.set_xlim(-6,6)
    ax.set_xlabel('ΔIL-2 induction')
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches="tight")
    plt.close()
    s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return '<img align="left" src="data:image/png;base64,%s">' % s

def strippplot_base64_image_selectivity_index_1mM(dist, dist2):
    """
    Plot distribution as stripplot and save the resulting image as a base64 image. 
    """
    sns.set(rc={'figure.figsize': (3, 1)})
    sns.set_style("ticks")
    sns.despine()

    df = pd.DataFrame(
        {'data': dist,
        'recent': dist2,
        })
    df = df.sort_values(by=['recent'], ascending=False)

    palette = sns.color_palette(['rebeccapurple', 'limegreen'])

    ax = sns.stripplot(data = df, x=df.data, hue=df.recent, palette=palette, legend=False, alpha = 0.9)
    ax.axvline(0,ls="--",c="red")
    ax.set_xlim(-2000,2000)
    ax.set_xlabel('Δselectivity index (1 mM ATP)')
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches="tight")
    plt.close()
    s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return '<img align="left" src="data:image/png;base64,%s">' % s

def find_examples_pic50(delta_df, query_idx):
    example_list = []
    for idx,row in delta_df.query("idx == @query_idx").sort_values("Delta",ascending=False).iterrows():
        smi_1, name_1, pic50_1, entry_date_1 = row.SMILES_1, row.Name_1, row.pic50_1, row.entry_date_1
        smi_2, name_2, pic50_2, entry_date_2 = row.SMILES_2, row.Name_2, row.pic50_2, row.entry_date_2
        tmp_list = [(smi_1, name_1, pic50_1, entry_date_1),(smi_2, name_2, pic50_2, entry_date_2)]
        tmp_list.sort(key=itemgetter(0))
        example_list.append(tmp_list[0])
        example_list.append(tmp_list[1])
    example_df = pd.DataFrame(example_list,columns=["SMILES","Name","pic50", 'entry_date'])
    return example_df

def find_examples_LogD(delta_df, query_idx):
    example_list = []
    for idx,row in delta_df.query("idx == @query_idx").sort_values("Delta",ascending=False).iterrows():
        smi_1, name_1, LogD_1 = row.SMILES_1, row.Name_1, row.LogD_1
        smi_2, name_2, LogD_2 = row.SMILES_2, row.Name_2, row.LogD_2
        tmp_list = [(smi_1, name_1, LogD_1),(smi_2, name_2, LogD_2)]
        tmp_list.sort(key=itemgetter(0))
        example_list.append(tmp_list[0])
        example_list.append(tmp_list[1])
    example_df = pd.DataFrame(example_list,columns=["SMILES","Name","LogD"])
    return example_df

def find_examples_MetStab(delta_df, query_idx):
    example_list = []
    for idx,row in delta_df.query("idx == @query_idx").sort_values("Delta",ascending=False).iterrows():
        smi_1, name_1, MetStab_1 = row.SMILES_1, row.Name_1, row.MetStab_1
        smi_2, name_2, MetStab_2 = row.SMILES_2, row.Name_2, row.MetStab_2
        tmp_list = [(smi_1, name_1, MetStab_1),(smi_2, name_2, MetStab_2)]
        tmp_list.sort(key=itemgetter(0))
        example_list.append(tmp_list[0])
        example_list.append(tmp_list[1])
    example_df = pd.DataFrame(example_list,columns=["SMILES","Name","MetStab"])
    return example_df

def find_examples_MAP4K3_HPK1(delta_df, query_idx):
    example_list = []
    for idx,row in delta_df.query("idx == @query_idx").sort_values("Delta",ascending=False).iterrows():
        smi_1, name_1, MAP4K3_HPK1_1 = row.SMILES_1, row.Name_1, row.MAP4K3_HPK1_1
        smi_2, name_2, MAP4K3_HPK1_2 = row.SMILES_2, row.Name_2, row.MAP4K3_HPK1_2
        tmp_list = [(smi_1, name_1, MAP4K3_HPK1_1),(smi_2, name_2, MAP4K3_HPK1_2)]
        tmp_list.sort(key=itemgetter(0))
        example_list.append(tmp_list[0])
        example_list.append(tmp_list[1])
    example_df = pd.DataFrame(example_list,columns=["SMILES","Name","MAP4K3_HPK1"])
    return example_df

def find_examples_selectivity_index(delta_df, query_idx):
    example_list = []
    for idx,row in delta_df.query("idx == @query_idx").sort_values("Delta",ascending=False).iterrows():
        smi_1, name_1, selectivity_index_1, delivery_date_1 = row.SMILES_1, row.Name_1, row.selectivity_index_1, row.delivery_date_1
        smi_2, name_2, selectivity_index_2, delivery_date_2  = row.SMILES_2, row.Name_2, row.selectivity_index_2, row.delivery_date_2
        tmp_list = [(smi_1, name_1, selectivity_index_1, delivery_date_1),(smi_2, name_2, selectivity_index_2, delivery_date_2)]
        tmp_list.sort(key=itemgetter(0))
        example_list.append(tmp_list[0])
        example_list.append(tmp_list[1])
    example_df = pd.DataFrame(example_list,columns=["SMILES","Name","selectivity_index", 'delivery_date'])
    return example_df

def find_examples_pSLP76(delta_df, query_idx):
    example_list = []
    for idx,row in delta_df.query("idx == @query_idx").sort_values("Delta",ascending=False).iterrows():
        smi_1, name_1, pSLP76_pIC50_1, delivery_date_1 = row.SMILES_1, row.Name_1, row.pSLP76_pIC50_1, row.delivery_date_1
        smi_2, name_2, pSLP76_pIC50_2, delivery_date_2  = row.SMILES_2, row.Name_2, row.pSLP76_pIC50_2, row.delivery_date_2
        tmp_list = [(smi_1, name_1, pSLP76_pIC50_1, delivery_date_1),(smi_2, name_2, pSLP76_pIC50_2, delivery_date_2)]
        tmp_list.sort(key=itemgetter(0))
        example_list.append(tmp_list[0])
        example_list.append(tmp_list[1])
    example_df = pd.DataFrame(example_list,columns=["SMILES","Name","pSLP76_pIC50", 'delivery_date'])
    return example_df

def find_examples_il2(delta_df, query_idx):
    example_list = []
    for idx,row in delta_df.query("idx == @query_idx").sort_values("Delta",ascending=False).iterrows():
        smi_1, name_1, il2_induction_1, upload_date_1 = row.SMILES_1, row.Name_1, row.il2_induction_1, row.upload_date_1
        smi_2, name_2, il2_induction_2, upload_date_2  = row.SMILES_2, row.Name_2, row.il2_induction_2, row.upload_date_2
        tmp_list = [(smi_1, name_1, il2_induction_1, upload_date_1),(smi_2, name_2, il2_induction_2, upload_date_2)]
        tmp_list.sort(key=itemgetter(0))
        example_list.append(tmp_list[0])
        example_list.append(tmp_list[1])
    example_df = pd.DataFrame(example_list,columns=["SMILES","Name","il2_induction", 'upload_date'])
    return example_df

def find_examples_selectivity_index_1mM(delta_df, query_idx):
    example_list = []
    for idx,row in delta_df.query("idx == @query_idx").sort_values("Delta",ascending=False).iterrows():
        smi_1, name_1, selectivity_index_1mM_1, upload_date_1 = row.SMILES_1, row.Name_1, row.selectivity_index_1mM_1, row.upload_date_1
        smi_2, name_2, selectivity_index_1mM_2, upload_date_2  = row.SMILES_2, row.Name_2, row.selectivity_index_1mM_2, row.upload_date_2
        tmp_list = [(smi_1, name_1, selectivity_index_1mM_1, upload_date_1),(smi_2, name_2, selectivity_index_1mM_2, upload_date_2)]
        tmp_list.sort(key=itemgetter(0))
        example_list.append(tmp_list[0])
        example_list.append(tmp_list[1])
    example_df = pd.DataFrame(example_list,columns=["SMILES","Name","il2_induction", 'upload_date'])
    return example_df