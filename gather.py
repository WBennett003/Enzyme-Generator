from utils import process_mols, AminoAcid_tokenise
import requests

def process_html_equation(html_equation, equation):
    chebi_equation = ''

    half_split = equation.split(' = ')
    reactants = half_split[0].split(' + ')
    products = half_split[1].split(' + ')
    n_reactants = len(reactants)
    n_products = len(products)

    splices = html_equation.split('molid=\"chebi:')
    reactant_ids = []
    product_ids = []

    for i, cut in enumerate(splices[1:]):
        idx = cut.index('"') #the first char after the id is / but this causes an error so i will -1 from the position of "
        id = cut[:idx]
        
        if i == n_reactants - 1:
            chebi_equation += id + ' = '
        elif i == len(splices) - 2:
            chebi_equation += id
        else:
            chebi_equation += id + ' + '
        
        
        if i < n_reactants:
            reactant_ids.append(id)
        else:
            product_ids.append(id)
    
    #convert to string to make sample data tabular
    # reactant_ids = ' '.join(reactant_ids)
    # product_ids = ' '.join(product_ids)


    return reactant_ids, product_ids, chebi_equation

def get_rhea(rhea_id, url='https://www.rhea-db.org/rhea/?query='):
    try:
        data = dict(requests.get(url+rhea_id[5:]+'&format=json').json())['results']
        if len(data) > 0:
            data = data[0]
            equation = data['equation']
            html_equation = data['htmlequation']
            reactants, products, chebi_equation = process_html_equation(html_equation, equation)
            return equation, chebi_equation, reactants, products
        return '', '', '', ''
    except:
        return '', '', '', ''


def get_enzyme(data, id):
    data = dict(data.json())['results']
    count = 0
    for sample in data:
        accension = sample['primaryAccession']
        uniprot_id = sample['uniProtkbId']
        sequence = sample['sequence']['value']
        seq_len = sample['sequence']['length']
        tax_id = sample['organism']['taxonId']
        try:
            name = sample['proteinDescription']['recommendedName']['fullName']['value']
        except:
            name = 'N/A'

        equation, chebi_equation, reactants, products = '', '', '', '' #to check if cross ref is present
        if 'comments' in sample.keys():
            for comms in sample['comments']:
                if comms['commentType'] == 'CATALYTIC ACTIVITY':
                    if 'ecNumber' in comms['reaction'].keys():
                        ec_id = comms['reaction']['ecNumber']
                    elif 'ecNumbers' in sample['proteinDescription']['recommendedName'].keys():
                        ec_id = sample['proteinDescription']['recommendedName']['ecNumbers'][0]['value']
                    else:
                        ec_id = '0.0.0.0'
                    if 'reactionCrossReferences' in comms['reaction'].keys():
                        for db in comms['reaction']['reactionCrossReferences']:
                            if db['database'] == 'Rhea':
                                Rhea_id = db['id']
                                equation, chebi_equation, reactants, products = get_rhea(Rhea_id)
                                reactants_smiles, products_smiles, smile_reaction = process_mols(reactants, products)

        if equation != '' and  smile_reaction != '':
            row = [
                accension, uniprot_id, sequence, str(seq_len), str(tax_id), name, ec_id, equation, chebi_equation, smile_reaction, reactants_smiles, products_smiles, Rhea_id
            ]
            row = '\t'.join(row)
            with open(f'datasets/shotgun/parellel_uniprot{id}.tsv', 'a+') as f:
                f.write(row+'\n')
            count += 1
        
    return count