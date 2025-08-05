import io
import base64
import os

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.TemplateAlign import AlignMolToTemplate2D

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

import numpy as np

var_list = ['pIC50',
            'mw',
            'tpsa',
            'num_rot_bonds',
            'hba',
            'hbd',
            'fraction_csp3',
            'logp'
            ]

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])

    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def scatterplot_confidence_ellipses_mw_base64_image(dist1, dist2, pic50_pred, mw):
    """
    Plot scatterplot with confidence ellipses of pIC50 and MW. 
    """
    fig, ax_nstd = plt.subplots(figsize=(3, 3))

    x = dist1
    y = dist2

    mu = dist1.mean(), dist2.mean()

    ax_nstd.axvline(c='grey', lw=1, x=mu[0])
    ax_nstd.axhline(c='grey', lw=1, y=mu[1])

    ax_nstd.scatter(x=x, y=y, s=2)

    confidence_ellipse(x, y, ax_nstd, n_std=1,
                    label=r'$1\sigma$', edgecolor='firebrick')
    confidence_ellipse(x, y, ax_nstd, n_std=2,
                    label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
    confidence_ellipse(x, y, ax_nstd, n_std=3,
                    label=r'$3\sigma$', edgecolor='blue', linestyle=':')

    ax_nstd.scatter(mu[0], mu[1], c='red', s=3)
    ax_nstd.scatter(mw, pic50_pred, c='red', s=50)
    ax_nstd.set_title('MW vs pIC50')
    plt.ylabel(var_list[0],fontsize=15)
    plt.xlabel(var_list[1],fontsize=15)
    plt.tick_params(labelsize=15)
    ax_nstd.legend()
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches="tight")
    plt.close()
    s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return '<img align="left" src="data:image/png;base64,%s">' % s

def scatterplot_confidence_ellipses_tpsa_base64_image(dist1, dist2, pic50_pred, tpsa):
    """
    Plot scatterplot with confidence ellipses of pIC50 and TPSA. 
    """
    fig, ax_nstd = plt.subplots(figsize=(4, 4))

    x = dist1
    y = dist2

    mu = dist1.mean(), dist2.mean()

    ax_nstd.axvline(c='grey', lw=1, x=mu[0])
    ax_nstd.axhline(c='grey', lw=1, y=mu[1])

    ax_nstd.scatter(x=x, y=y, s=2)

    confidence_ellipse(x, y, ax_nstd, n_std=1,
                    label=r'$1\sigma$', edgecolor='firebrick')
    confidence_ellipse(x, y, ax_nstd, n_std=2,
                    label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
    confidence_ellipse(x, y, ax_nstd, n_std=3,
                    label=r'$3\sigma$', edgecolor='blue', linestyle=':')

    ax_nstd.scatter(mu[0], mu[1], c='red', s=3)
    ax_nstd.scatter(tpsa, pic50_pred, c='red', s=50)
    ax_nstd.set_title('TPSA vs pIC50')
    plt.ylabel(var_list[0],fontsize=15)
    plt.xlabel(var_list[2],fontsize=15)
    plt.tick_params(labelsize=15)
    ax_nstd.set(xlim=(20, 185)) # manually set range
    ax_nstd.legend()
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches="tight")
    plt.close()
    s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return '<img align="left" src="data:image/png;base64,%s">' % s

def scatterplot_confidence_ellipses_logp_base64_image(dist1, dist2, pic50_pred, logp):
    """
    Plot scatterplot with confidence ellipses of pIC50 and logp. 
    """
    fig, ax_nstd = plt.subplots(figsize=(4, 4))

    x = dist1
    y = dist2

    mu = dist1.mean(), dist2.mean()

    ax_nstd.axvline(c='grey', lw=1, x=mu[0])
    ax_nstd.axhline(c='grey', lw=1, y=mu[1])

    ax_nstd.scatter(x=x, y=y, s=2)

    confidence_ellipse(x, y, ax_nstd, n_std=1,
                    label=r'$1\sigma$', edgecolor='firebrick')
    confidence_ellipse(x, y, ax_nstd, n_std=2,
                    label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
    confidence_ellipse(x, y, ax_nstd, n_std=3,
                    label=r'$3\sigma$', edgecolor='blue', linestyle=':')

    ax_nstd.scatter(mu[0], mu[1], c='red', s=3)
    ax_nstd.scatter(logp, pic50_pred, c='red', s=50)
    ax_nstd.set_title('logp vs pIC50')
    plt.ylabel(var_list[0],fontsize=15)
    plt.xlabel(var_list[7],fontsize=15)
    plt.tick_params(labelsize=15)
    ax_nstd.legend()
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches="tight")
    plt.close()
    s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return '<img align="left" src="data:image/png;base64,%s">' % s

def scatterplot_confidence_ellipses_hbd_base64_image(dist1, dist2, pic50_pred, hbd):
    """
    Plot scatterplot with confidence ellipses of pIC50 and logp. 
    """
    fig, ax_nstd = plt.subplots(figsize=(4, 4))

    x = dist1
    y = dist2

    mu = dist1.mean(), dist2.mean()

    ax_nstd.axvline(c='grey', lw=1, x=mu[0])
    ax_nstd.axhline(c='grey', lw=1, y=mu[1])

    ax_nstd.scatter(x=x, y=y, s=2)

    confidence_ellipse(x, y, ax_nstd, n_std=1,
                    label=r'$1\sigma$', edgecolor='firebrick')
    confidence_ellipse(x, y, ax_nstd, n_std=2,
                    label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
    confidence_ellipse(x, y, ax_nstd, n_std=3,
                    label=r'$3\sigma$', edgecolor='blue', linestyle=':')

    ax_nstd.scatter(mu[0], mu[1], c='red', s=3)
    ax_nstd.scatter(hbd, pic50_pred, c='red', s=50)
    ax_nstd.set_title('HBD vs pIC50')
    plt.ylabel(var_list[0],fontsize=15)
    plt.xlabel(var_list[5],fontsize=15)
    plt.tick_params(labelsize=15)
    ax_nstd.legend()
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches="tight")
    plt.close()
    s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return '<img align="left" src="data:image/png;base64,%s">' % s

def scatterplot_confidence_ellipses_hba_base64_image(dist1, dist2, pic50_pred, hba):
    """
    Plot scatterplot with confidence ellipses of pIC50 and logp. 
    """
    fig, ax_nstd = plt.subplots(figsize=(4, 4))

    x = dist1
    y = dist2

    mu = dist1.mean(), dist2.mean()

    ax_nstd.axvline(c='grey', lw=1, x=mu[0])
    ax_nstd.axhline(c='grey', lw=1, y=mu[1])

    ax_nstd.scatter(x=x, y=y, s=2)

    confidence_ellipse(x, y, ax_nstd, n_std=1,
                    label=r'$1\sigma$', edgecolor='firebrick')
    confidence_ellipse(x, y, ax_nstd, n_std=2,
                    label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
    confidence_ellipse(x, y, ax_nstd, n_std=3,
                    label=r'$3\sigma$', edgecolor='blue', linestyle=':')

    ax_nstd.scatter(mu[0], mu[1], c='red', s=3)
    ax_nstd.scatter(hba, pic50_pred, c='red', s=50)
    ax_nstd.set_title('HBA vs pIC50')
    plt.ylabel(var_list[0],fontsize=15)
    plt.xlabel(var_list[4],fontsize=15)
    plt.tick_params(labelsize=15)
    ax_nstd.legend()
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches="tight")
    plt.close()
    s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return '<img align="left" src="data:image/png;base64,%s">' % s

def scatterplot_confidence_ellipses_num_rot_bonds_base64_image(dist1, dist2, pic50_pred, num_rot_bonds):
    """
    Plot scatterplot with confidence ellipses of pIC50 and logp. 
    """
    fig, ax_nstd = plt.subplots(figsize=(3.5, 3.5))

    x = dist1
    y = dist2

    mu = dist1.mean(), dist2.mean()

    ax_nstd.axvline(c='grey', lw=1, x=mu[0])
    ax_nstd.axhline(c='grey', lw=1, y=mu[1])

    ax_nstd.scatter(x=x, y=y, s=2)

    confidence_ellipse(x, y, ax_nstd, n_std=1,
                    label=r'$1\sigma$', edgecolor='firebrick')
    confidence_ellipse(x, y, ax_nstd, n_std=2,
                    label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
    confidence_ellipse(x, y, ax_nstd, n_std=3,
                    label=r'$3\sigma$', edgecolor='blue', linestyle=':')

    ax_nstd.scatter(mu[0], mu[1], c='red', s=3)
    ax_nstd.scatter(num_rot_bonds, pic50_pred, c='red', s=50)
    ax_nstd.set_title('num_rot_bonds vs pIC50')
    plt.ylabel(var_list[0],fontsize=15)
    plt.xlabel(var_list[3],fontsize=15)
    plt.tick_params(labelsize=15)
    ax_nstd.legend()
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches="tight")
    plt.close()
    s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return '<img align="left" src="data:image/png;base64,%s">' % s

def scatterplot_confidence_ellipses_fraction_csp3_base64_image(dist1, dist2, pic50_pred, fraction_csp3):
    """
    Plot scatterplot with confidence ellipses of pIC50 and logp. 
    """
    fig, ax_nstd = plt.subplots(figsize=(4, 4))

    x = dist1
    y = dist2

    mu = dist1.mean(), dist2.mean()

    ax_nstd.axvline(c='grey', lw=1, x=mu[0])
    ax_nstd.axhline(c='grey', lw=1, y=mu[1])

    ax_nstd.scatter(x=x, y=y, s=2)

    confidence_ellipse(x, y, ax_nstd, n_std=1,
                    label=r'$1\sigma$', edgecolor='firebrick')
    confidence_ellipse(x, y, ax_nstd, n_std=2,
                    label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
    confidence_ellipse(x, y, ax_nstd, n_std=3,
                    label=r'$3\sigma$', edgecolor='blue', linestyle=':')

    ax_nstd.scatter(mu[0], mu[1], c='red', s=3)
    ax_nstd.scatter(fraction_csp3, pic50_pred, c='red', s=50)
    ax_nstd.set_title('fraction_csp3 vs pIC50')
    plt.ylabel(var_list[0],fontsize=15)
    plt.xlabel(var_list[6],fontsize=15)
    plt.tick_params(labelsize=15)
    ax_nstd.legend()
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches="tight")
    plt.close()
    s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return '<img align="left" src="data:image/png;base64,%s">' % s

def mol_to_base64_image(mol, Molecule):
    """
    Convert RDKit mol to an image
    """

    templates_path = "/home/ubuntu/projects/DD_projects/project-akka/dev-corey/_refs/scaffold_templates/"
    templates = os.listdir(templates_path)
    
    drawer = rdMolDraw2D.MolDraw2DCairo(800, 400)
    opts = drawer.drawOptions()
    opts.fontFile = f'{templates_path}//Roboto/Roboto-Medium.ttf'
    opts.clearBackground = True
    opts.padding = 0.1
    opts.legendFontSize = 50
    
    # scaffold template to align to for isoquin-indazole series

    for template in templates:
        try:
            template = "/".join([templates_path, template])
            template = Chem.MolFromMolFile(template)
            AlignMolToTemplate2D(mol, template)
            mol.__sssAtoms = []
            break
        except ValueError:
            pass
    else:
        smiles = Chem.MolToSmiles(mol)
        raise ValueError(f"No matching template for {smiles}")

    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, legend=Molecule)
    drawer.FinishDrawing()
    bio = io.BytesIO()
    text = drawer.GetDrawingText()
    im_text64 = base64.b64encode(text).decode('utf8')
    img_str = f"<img src='data:image/png;base64, {im_text64}'/>"
    return img_str