import open_clip
import glob
from collections import OrderedDict
import torch
from PIL import Image
import open_clip
import urllib.request
from io import BytesIO

#model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub_microsoft_pretrained_high', pretrained='/admin/home-thieme/alex/open_clip/best_model.pt')
#model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub_microsoft', pretrained='/admin/home-thieme/log/pubmedbert_unlocked_4_layers_40e_5lr_aug_multinode_v5/checkpoints/epoch_40.pt')
tokenizer = open_clip.get_tokenizer('hf-hub_microsoft_pretrained_high')


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

context_length = 256

def classify_image_url(image_url, labels, template):
    print(image_url)
    with urllib.request.urlopen(image_url) as url:
        image = Image.open(BytesIO(url.read()))

    cleaned_labels = []
    ground_truth = None
    for label in labels:
        if label.startswith('*'):
            ground_truth = label[1:]
            cleaned_labels.append(ground_truth)
        else:
            cleaned_labels.append(label)

    image_tensor = preprocess_val(image).unsqueeze(0).to(device)
    texts = tokenizer([template + l for l in cleaned_labels], context_length=context_length).to(device)

    with torch.no_grad():
        image_features, text_features, logit_scale = model(image_tensor, texts)
        logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
        sorted_indices = torch.argsort(logits, dim=-1, descending=True)

    top_prediction_index = sorted_indices[0][0]
    return cleaned_labels[top_prediction_index] == ground_truth

# Example usage
images = [
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20230420&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            'assault',
            'fall',
            'malignancy',
            'osteoporosis',
            '*repetitive strain'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20140626&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            'copper',
            '*thiamine',
            'vitamin B6',
            'vitamin C',
            'zinc'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20210909&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            'angiosarcoma',
            'bacillary angiomatosis',
            'herpes simplex keratoconjunctivitis',
            '*kaposi sarcoma',
            'ocular surface squamous neoplasia'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20140821&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            'Diaphragmatic hernia',
            'Gastric bezoar',
            'Inferior mesenteric artery thrombosis',
            'Pancreatic phlegmon',
            '*Small-bowel volvulus'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20191205&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            '*Hydrocolpos',
            'Prolapsed uterus',
            'Interlabial cyst',
            'McKusick-Kaufman syndrome',
            'Rhabdomyosarcoma'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20181101&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            'Scurvy',
            'Bismuth poisoning',
            'Adults Still’s disease',
            'Behçet’s syndrome',
            '*Lead poisoning'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20170803&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            'Abscess',
            'Dermatofibrosarcoma protuberans',
            'Hematoma',
            '*Insulin-induced lipohypertrophy',
            'Nodular fasciitis'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20100729&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            '*Alkaline phosphatase',
            '{beta}2-microglobulin',
            'Cortisol',
            'Insulin-like growth factor 1',
            'Mean corpuscular volume'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20160915&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            'Wuchereria bancrofti',
            'Onchocerca volvulus',
            'Borrelia recurrentis',
            'Trypanosoma brucei',
            '*Plamodium vivax'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20130620&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            'Angioedema',
            'Contact dermatitis',
            'Follicular lymphoma',
            'Hypothyroidism',
            '*Sjögrens syndrome'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20131107&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            '*Cholesterol',
            'Calcium apatite',
            'Calcium oxalate',
            'Calcium pyrophosphate dihydrate',
            'Monosodium urate'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20111006&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            'Acromegaly',
            'Cushings disease',
            '*Graves disease',
            'Hashimotos thyroiditis',
            'Type 1 diabetes'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20180920&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            'Herpes simplex virus-1 gingivostomatitis',
            'Stevens–Johnson syndrome',
            'Behçet’s disease',
            'Mucocutaneous Epstein–Barr virus',
            '*Mycoplasma pneumoniae-associated mucositis'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20060525&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            '*Palsy of the long thoracic nerve',
            'Subcutaneous emphysema',
            'Neuralgic amyotrophy',
            'Pleural prolapse',
            'Scapular subluxation'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20180111&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            '*Bordetella pertussis',
            'Chlamydial pneumonia',
            'Mycoplasmal pneumonia',
            'Bronchiolitis',
            'Respiratory syncytial virus infection'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20080724&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            'Amyloidosis',
            '*Celiac disease',
            'Hypothyroidism',
            'Kawasaki disease',
            'Type 2 diabetes'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20181129&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            'Saccular cyst',
            '*Laryngocele',
            'Thyroglossal cyst',
            'Scrofula',
            'Lymphadenopathy'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20220616&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            'Cataract',
            'Coats disease',
            'Ocular toxocariasis',
            'Retinal detachment',
            '*Retinoblastoma'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20070726&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            'Erythrasma',
            'Recurrent breast cancer',
            'Radiation dermatitis',
            '*Cellulitis',
            'Lymphedema'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20180621&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            'Superior thyroid artery thrombosis',
            'Ascending pharyngeal artery thrombosis',
            'Posterior auricular artery thrombosis',
            'Facial artery thrombosis',
            '*Lingual artery thrombosis'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20100916&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            'Acute myelogenous leukemia',
            'Ehrlichiosis',
            '*Lead poisoning',
            'Malaria',
            'Pompe disease'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20051229&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            'Aspergillosis',
            'Adrenal insufficiency',
            'Oral leukoplakia',
            'Pellagra',
            '*Lingua villosa nigra'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20051222&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            'Addisons disease',
            '*Insulin resistance',
            'Growth hormone excess',
            'Glucagonoma',
            'Diabetes insipidus'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20071101&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            'Ketamine abuse',
            'Heroin abuse',
            '*Cocaine abuse',
            'Phencyclidine abuse',
            'Mescaline abuse'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20110707&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            'Esophageal rupture',
            'Flail chest',
            '*Hydropneumothorax',
            'Lymphangioleiomyomatosis',
            'Phrenic nerve palsy'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20130425&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            'Aortic regurgitation',
            'Chvostek sign',
            '*Lisch nodules',
            'Thyroid bruit',
            'Web neck'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20181206&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            'Chronic lung allograft dysfunction',
            'Acute promyelocytic leukemia',
            'Adverse reaction to an immunosuppressant',
            '*Parvovirus B19 infection',
            'Acute hepatitis B infection'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20140501&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            'Aortic dissection',
            'Cardiac tamponade',
            'Diffuse pulmonary hemorrhage',
            '*Tension pneumothorax',
            'Traumatic diaphragmatic hernia'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20100930&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            'Brachial plexus injury',
            '*Erythromelalgia',
            'Frostbite',
            'Raynaud phenomenon',
            'Subclavian artery stenosis'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20180830&width=1500&height=4000",
        "template": "this is a photo of ",
        "labels": [
            '*Tetracyclines',
            'Penicillins',
            'Macrolides',
            'Fluoroquinolones',
            'Sulfonamides'
        ]
    },
    {
        "url": "",
        "template": "this is a photo of ",
        "labels": [
            '',
            '',
            '',
            '',
            ''
        ]
    },
]

total_predictions = 0
correct_predictions = 0

for image in images:
    if(image["url"] != ""):
	    if classify_image_url(image["url"], image["labels"], image["template"]):
                print("correct!")
                correct_predictions += 1
	    total_predictions += 1

accuracy = correct_predictions / total_predictions * 100
print(f"Accuracy: {accuracy:.2f}%")
