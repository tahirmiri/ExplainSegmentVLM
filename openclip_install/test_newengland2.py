import open_clip
import glob
from collections import OrderedDict
import torch
from PIL import Image
import open_clip
import urllib.request
from io import BytesIO

#model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub_microsoft', pretrained='/admin/home-thieme/alex/open_clip/best_model.pt')
#model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub_microsoft', pretrained='/admin/home-thieme/log/pubmedbert_unlocked_4_layers_40e_5lr_aug_multinode_v5/checkpoints/epoch_40.pt')
tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')


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
        "template": "A 60-year-old left-handed woman presented to the emergency department with pain in her left forearm. The arm was swollen and tender, especially with passive pronation and supination. The overlying skin was intact, and the results of neurovascular examination were normal. Radiographs of the left forearm were performed (upper image, anteroposterior view; lower image, lateral view). The findings should raise concern for which contributory factor? This is an x-ray of ",
        "labels": [
            '*assault',
            'fall',
            'malignancy',
            'osteoporosis',
            'repetitive strain'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20140626&width=1500&height=4000",
        "template": "Deficiency of what micronutrient is likely to account for this rash following Roux-en-Y gastric bypass? This is a photo of ",
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
        "template": "A 50-year-old man presented with a lesion on his eye that developed over the preceding month. He also had violaceous plaques on his back and lower limbs. Testing for HIV was positive. What is the diagnosis? This is a photo of ",
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
        "template": "What is the diagnosis? This is a CT of ",
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
        "template": "Physical examination of a 1-day-old girl born at 36 weeks of gestation revealed a soft mass protruding from the external genitalia. The mass was noted to increase in size when the infant cried. The physical examination was otherwise normal. What is the diagnosis? This is a photo of ",
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
        "template": "A 39-year-old man presented to the emergency department with a 4-week history of abdominal pain and constipation. Physical examination of the abdomen was normal, but he was noted to have gray lines along the margins of his lower gum. What is the most likely diagnosis? This is a photo of ",
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
        "template": "What is the most likely diagnosis for these two painless periumbilical masses in a 76-year-old man with type 2 diabetes mellitus? This is a photo of ",
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
        "template": "Which laboratory measure would be expected to be most abnormal in this patient? This is a x-ray of ",
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
        "template": "An Eritrean man presented with 3 years of intermittent fevers; workup revealed anemia, elevated aminotransferase levels, and this finding on the blood smear. What organism caused these symptoms? This is a microscopy of ",
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
        "template": "This patient presented with xerostomia and xerophthalmia. What is the diagnosis? This is a photo of ",
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
        "template": "What are these crystals that were aspirated from the bursa of an elbow of a patient with rheumatoid arthritis? This is a microscopy of ",
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
        "template": "This patient was being treated by an endocrinologist for which one of the following conditions? This is a photo of ",
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
        "template": "A 26-year-old previously healthy man presented to the emergency department with a 3-day history of fever, dry cough, and nonpruritic rash. A physical examination was notable for crackles on the left side of the chest and a macular, targetoid rash on his hands and feet, including the palms and soles. Over the next 3 days, severe mucositis developed that involved the lips, buccal mucosa, conjunctivae, and urethral meatus. What is the diagnosis? This is a photo of ",
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
        "template": "This 9-year-old boy presented with a two-day history of right shoulder pain after an upper respiratory tract infection. What is the cause of the abnormality demonstrated? This is a photo of ",
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
        "template": "A 66-year-old woman presented with a 2-week history of dry cough and severe pain in the right flank. Five days before presentation, she had received a diagnosis of viral upper respiratory tract infection, but her symptoms had not abated with supportive treatment. The patient reported no trauma; she had no known sick contacts and had received the tetanus–diphtheria–acellular pertussis vaccine 8 years earlier. Physical examination revealed tenderness on palpation over the chest wall on the right side. Computed tomography of the chest and abdomen revealed a displaced fracture of the lateral aspect of the ninth rib on the right side. What is the likely diagnosis? This is a CT of ",
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
        "template": "What is the most likely diagnosis? This is a photo of ",
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
        "template": "A 58-year-old man presented to an outpatient clinic with a 2-year history of progressive hoarseness and swelling on the left side of his neck. He worked as a farmer with no history of tobacco use. Physical examination was remarkable for nontender, compressible swelling in the left cervical region. What is the most likely diagnosis? This is a photo of ",
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
        "template": "A 3-year-old girl was brought to the emergency department with a 2-month history of a white pupil and a 1-day history of redness and pain in the right eye. An eye examination showed leukocoria, as well as iris neovascularization and a white, nodular mass in the posterior chamber. The left eye was normal. B-scan ultrasonography showed calcification of the mass and vitreous seeding in the affected eye. Which of the following is the most likely diagnosis? This is a photo and MRI of ",
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
        "template": "This patient presented with acute onset of erythema. Nearly a year earlier she was treated for invasive breast cancer. What is the diagnosis? This is a photo of ",
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
        "template": "An 86-year-old woman presented to the emergency department with tongue pain 8 days after the diagnosis of giant-cell arteritis by temporal artery biopsy and treatment with glucocorticoids. Examination revealed necrotic ulceration on the right side of the tongue. Cervicofacial CT showed complete thrombosis of which one of the following arteries, on the right side? This is a photo of ",
        "labels": [
            'Superior thyroid artery',
            'Ascending pharyngeal artery',
            'Posterior auricular artery',
            'Facial artery',
            '*Lingual artery'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20100916&width=1500&height=4000",
        "template": "What is the diagnosis? This is a microscopy of ",
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
        "template": "What is the diagnosis? This is a photo of ",
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
        "template": "What endocrinopathy is most frequently associated with this sign? This is a photo of ",
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
        "template": "Which one of the following drugs of abuse is most typically associated with the illustrated complication? This is a photo of ",
        "labels": [
            'Ketamine',
            'Heroin',
            '*Cocaine',
            'Phencyclidine',
            'Mescaline'
        ]
    },
    {
        "url": "https://csvc.nejm.org/ContentServer/images?id=IC20110707&width=1500&height=4000",
        "template": "What is the diagnosis? This is a x-ray of ",
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
        "template": "In addition to neurofibromatosis, what other examination finding would you expect for this patient? This is a photo of ",
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
        "template": "A 67-year-old woman presented to the emergency department with a 6-week history of progressive exertional dyspnea. Her medical history was notable for lung transplantation that had been performed 8 years earlier. Laboratory studies showed normocytic anemia, with a hemoglobin level of 6.9 g per deciliter (reference range, 11.9 to 17.2). White-cell and platelet counts were normal. The reticulocyte index was 0%. Bone marrow aspiration was performed and showed giant proerythroblasts with basophilic and vacuolated cytoplasm, uncondensed chromatin, and large, intranuclear, purple-colored inclusions. What is the diagnosis? This is a microscopy of ",
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
        "template": "What is the diagnosis in this patient who had been involved in a motorcycle accident? This is a x-ray of ",
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
        "template": "Movement, bathing, and exercise triggered pain in this patient's right hand. What is the most likely diagnosis? This is a photo of ",
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
        "template": "A 55-year-old woman was admitted to hospital after sustaining a severe crush injury to both legs in a motor vehicle accident. A polymicrobial wound infection developed and she received antibiotic treatment. Black discoloration of her tongue was observed within 1 week of starting treatment, and the patient reported nausea and a bad taste in her mouth. Which class of antibiotic can lead to this presentation? This is a photo of ",
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
