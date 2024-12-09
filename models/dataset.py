import json
import math

from parsivar import *
from hazm import *
from models.Term import Term
import re


class Dataset:
    def __init__(self,dataset_path,data=None,preprocessedDocs=None):
        self.path = dataset_path
        self.data = data
        self.preprocessedDocs = preprocessedDocs
        self.pos_index = None
        self.docs_norm = None
        self.stop_words = []
        # self.idf_list = []
        # self.ttt = set()
        #self.tfff = []
        #self.tof = None

    def read_dataset(self):
        file = open(self.path)
        self.data = json.load(file)

    def get_fields(self):
        fields = []
        for field in list(self.data.values())[0]:
            fields.append(field)
        return fields

    def read_data_at_index(self,index):
        res = []
        for field in list(self.data.values())[index]:
            res.append([field,list(self.data.values())[index][field]])
        return res

    def get_titles(self):
        titles = []
        for d in self.data:
            titles.append(self.data[d]['title'])
        return titles

    def get_contents(self):
        contents = []
        for d in self.data:
            contents.append(self.data[d]['content'])
        return contents

    def get_urls(self):
        urls = []
        for d in self.data:
            urls.append(self.data[d]['url'])
        return urls

    def get_size(self):
        return len(self.data)

    def read_title_at_index(self,index):
        return self.get_titles()[index]

    def read_content_at_index(self,index):
        return self.get_contents()[index]

    def read_url_at_index(self,index):
        return self.get_urls()[index]

    def preprocess(self):
        normalizer = Normalizer()
        stemmer = FindStems()
        stopWords = ['.','?','/','،',':',')','(','»','«','[',']','{','}',',','؛', '*','#','=','+','-','$','!','@','%',
                          '^','&','_',';','~','`','"','\'','|','\\','<','>']
        copy_stopWords = []
        for s in stopWords:
            copy_stopWords.append(s)
        tokens_repeat = {}
        for content in self.get_contents():
            normalized_content = normalizer.normalize(content)
            content_tokens = normalized_content.strip().split()
            for ct in content_tokens:
                if len(ct.strip("\u200c"))!=0:
                    ct.strip("\u200c")
            content_tokens = [x.strip("\u200c") for x in content_tokens if len(x.strip("\u200c")) != 0]
            for i in range(len(content_tokens)):
                tmp = content_tokens[i]
                f = ""
                for j in range(len(tmp)):
                    if tmp[j] in stopWords:
                        continue
                    f+=str(tmp[j])
                content_tokens[i] = f

            for t in content_tokens:
                if t!="":
                    if tokens_repeat.get(t) is not None:
                        tokens_repeat[t] += 1
                    else:
                        tokens_repeat[t] = 1
        sorted_tokens = dict(sorted(tokens_repeat.items(), key=lambda x: x[1],reverse=True))
        stopWords.extend(list(sorted_tokens.keys())[:50])
        self.stop_words = stopWords
        #print(stopWords)
        docs = []
        for i,content in enumerate(self.get_contents()):
            normalized_content = normalizer.normalize(content)
            content_tokens = normalized_content.strip().split()
            for ct in content_tokens:
                if len(ct.strip("\u200c")) != 0:
                    ct.strip("\u200c")
            content_tokens = [x.strip("\u200c") for x in content_tokens if len(x.strip("\u200c")) != 0]
            new_tokens = []
            for p in range(len(content_tokens)):
                tmp = content_tokens[p]
                f = ""
                for j in range(len(tmp)):
                    if tmp[j] in copy_stopWords:
                        continue
                    f += str(tmp[j])
                if f!="":
                    new_tokens.append(f)
            tokens = []
            for token in new_tokens:
                token = stemmer.convert_to_stem(token)
                if token in stopWords:
                    continue
                # if '\u200c' in token:
                #     token = re.sub('\u200c'," ",token)
                tokens.append(token)
            docs.append(tokens)
        self.preprocessedDocs = docs
        return docs

    def positional_index(self):
        if self.preprocessedDocs is None:
            self.preprocess()
        pos_index = {}
        for d in range(len(self.preprocessedDocs)):
            for i in range(len(self.preprocessedDocs[d])):
                tmp = Term()
                if self.preprocessedDocs[d][i] in pos_index:
                    tmp = pos_index[self.preprocessedDocs[d][i]]
                if d not in tmp.position_in_docs:
                    tmp.position_in_docs[d] = []
                    tmp.frequency_in_docs[d] = 0
                tmp.frequency_in_docs[d] += 1
                tmp.total_frequency += 1
                tmp.position_in_docs[d].append(i)
                pos_index[self.preprocessedDocs[d][i]] = tmp
        self.pos_index = pos_index
        return pos_index

    def get_dictionary(self):
        if self.pos_index is None:
            self.positional_index()
        return self.pos_index.keys()

    def tf(self,term, doc_id):
        t = self.pos_index[term]
        if t.frequency_in_docs[doc_id] > 0:
            return 1 + math.log10(t.frequency_in_docs[doc_id])
        return 0

    def idf(self,term):
        t = self.pos_index[term]
        nt = len(t.frequency_in_docs.keys())
        return math.log10(self.get_size() / nt)

    def tf_idf(self,term, doc_id):
        return self.tf(term,doc_id) * self.idf(term)

    def calculate_weights(self):
        if self.pos_index is None:
            self.positional_index()
        for t in self.get_dictionary():
            postings = self.pos_index[t].frequency_in_docs
            for d in postings.keys():
                self.pos_index[t].weight_in_docs[d] = self.tf_idf(t,d)

    def norm_docs(self):
        self.docs_norm = [0] * (self.get_size())
        for t in self.get_dictionary():
            postings = self.pos_index[t].frequency_in_docs
            for d in postings.keys():
                self.docs_norm[d] += (self.pos_index[t].weight_in_docs[d] ** 2)
        for i,n in enumerate(self.docs_norm):
            self.docs_norm[i] = math.sqrt(n)

    # def no_norm_docs(self):
    #     self.docs_norm = [0] * (self.get_size())
    #     for t in self.get_dictionary():
    #         postings = self.pos_index[t].frequency_in_docs
    #         for d in postings.keys():
    #             self.docs_norm[d] += (self.pos_index[t].weight_in_docs[d] ** 2)


    def calculate_cosine_similarity(self,query):
        if self.docs_norm is None:
            self.norm_docs()
        scores = {}
        mapping = {}
        for q in query:
            if mapping.get(q) is not None:
                mapping[q] += 1
            else:
                mapping[q] = 1
        for w in query:
            #print(w)
            q_weight = mapping[w] * self.idf(w)
            #print(q_weight)
            for d in self.pos_index[w].weight_in_docs:
                weight = self.pos_index[w].weight_in_docs[d]
                if d not in scores and d is not None:
                    scores[d] = 0
                scores[d] += weight * q_weight
        for d in scores:
            scores[d] = scores[d] / self.docs_norm[d]
        # print(scores)
        sorted_scores = dict(sorted(scores.items(), key=lambda x:x[1],reverse=True))
        print(sorted_scores)
        return sorted_scores

    def k_nearest_documents(self,query,k):
        query = [query]
        normalizer = Normalizer()
        stemmer = FindStems()
        docs = []
        for content in query:
            normalized_content = normalizer.normalize(content)
            content_tokens = normalized_content.strip().split()
            for ct in content_tokens:
                if len(ct.strip("\u200c")) != 0:
                    ct.strip("\u200c")
            content_tokens = [x.strip("\u200c") for x in content_tokens if len(x.strip("\u200c")) != 0]
            tokens = []
            for token in content_tokens:
                token = stemmer.convert_to_stem(token)
                if token in self.stop_words:
                    continue
                tokens.append(token)
            docs.append(tokens)
        scores = self.calculate_cosine_similarity(docs[0])
        return list(scores)[:k]

    def create_champion_list(self,k):
        for t in self.pos_index.keys():
            sorted_scores = dict(sorted(self.pos_index[t].weight_in_docs.items(), key=lambda x: x[1], reverse=True))
            self.pos_index[t].champion_list = list(sorted_scores)[:k]

    def calculate_cosine_similarity_champion(self,query):
        # self.create_champion_list(300)
        if self.docs_norm is None:
            self.norm_docs()
        scores = {}
        mapping = {}
        for q in query:
            if mapping.get(q) is not None:
                mapping[q] += 1
            else:
                mapping[q] = 1
        for w in query:
            #print(w)
            q_weight = mapping[w] * self.idf(w)
            #print(q_weight)
            for d in self.pos_index[w].champion_list:
                weight = self.pos_index[w].weight_in_docs[d]
                if d not in scores and d is not None:
                    scores[d] = 0
                scores[d] += weight * q_weight
        for d in scores:
            scores[d] = scores[d] / self.docs_norm[d]
        #print(scores)
        sorted_scores = dict(sorted(scores.items(), key=lambda x:x[1],reverse=True))
        #print(sorted_scores)
        return sorted_scores

    def k_nearest_documents_champion(self,query,k):
        query = [query]
        #print(query)
        normalizer = Normalizer()
        stemmer = FindStems()
        docs = []
        for content in query:
            normalized_content = normalizer.normalize(content)
            content_tokens = normalized_content.strip().split()
            for ct in content_tokens:
                if len(ct.strip("\u200c")) != 0:
                    ct.strip("\u200c")
            content_tokens = [x.strip("\u200c") for x in content_tokens if len(x.strip("\u200c")) != 0]
            #print(content_tokens)
            #print(self.stop_words)
            tokens = []
            for token in content_tokens:
                token = stemmer.convert_to_stem(token)
                if token in self.stop_words:
                    continue
                tokens.append(token)
            docs.append(tokens)
        #print(docs)
        scores = self.calculate_cosine_similarity_champion(docs[0])
        return list(scores)[:k]

    def normalize(self):
        translation_src = "ؠػػؽؾؿكيٮٯٷٸٹٺٻټٽٿڀځٵٶٷٸٹٺٻټٽٿڀځڂڅڇڈډڊڋڌڍڎڏڐڑڒړڔڕږڗڙښڛڜڝڞڟڠڡڢڣڤڥڦڧڨڪګڬڭڮڰڱڲڳڴڵڶڷڸڹںڻڼڽھڿہۂۃۄۅۆۇۈۉۊۋۏۍێېۑےۓەۮۯۺۻۼۿݐݑݒݓݔݕݖݗݘݙݚݛݜݝݞݟݠݡݢݣݤݥݦݧݨݩݪݫݬݭݮݯݰݱݲݳݴݵݶݷݸݹݺݻݼݽݾݿࢠࢡࢢࢣࢤࢥࢦࢧࢨࢩࢪࢫࢮࢯࢰࢱࢬࢲࢳࢴࢶࢷࢸࢹࢺࢻࢼࢽﭐﭑﭒﭓﭔﭕﭖﭗﭘﭙﭚﭛﭜﭝﭞﭟﭠﭡﭢﭣﭤﭥﭦﭧﭨﭩﭮﭯﭰﭱﭲﭳﭴﭵﭶﭷﭸﭹﭺﭻﭼﭽﭾﭿﮀﮁﮂﮃﮄﮅﮆﮇﮈﮉﮊﮋﮌﮍﮎﮏﮐﮑﮒﮓﮔﮕﮖﮗﮘﮙﮚﮛﮜﮝﮞﮟﮠﮡﮢﮣﮤﮥﮦﮧﮨﮩﮪﮫﮬﮭﮮﮯﮰﮱﺀﺁﺃﺄﺅﺆﺇﺈﺉﺊﺋﺌﺍﺎﺏﺐﺑﺒﺕﺖﺗﺘﺙﺚﺛﺜﺝﺞﺟﺠﺡﺢﺣﺤﺥﺦﺧﺨﺩﺪﺫﺬﺭﺮﺯﺰﺱﺲﺳﺴﺵﺶﺷﺸﺹﺺﺻﺼﺽﺾﺿﻀﻁﻂﻃﻄﻅﻆﻇﻈﻉﻊﻋﻌﻍﻎﻏﻐﻑﻒﻓﻔﻕﻖﻗﻘﻙﻚﻛﻜﻝﻞﻟﻠﻡﻢﻣﻤﻥﻦﻧﻨﻩﻪﻫﻬﻭﻮﻯﻰﻱﻲﻳﻴىكي“” "
        translation_dst = (
            'یککیییکیبقویتتبتتتبحاوویتتبتتتبحححچدددددددددررررررررسسسصصطعففففففققکککککگگگگگللللنننننهچهههوووووووووییییییهدرشضغهبببببببححددرسعععففکککممنننلررسححسرحاایییووییحسسکببجطفقلمییرودصگویزعکبپتریفقنااببببپپپپببببتتتتتتتتتتتتففففححححححححچچچچچچچچددددددددژژررککککگگگگگگگگگگگگننننننههههههههههییییءاااووااییییااببببتتتتثثثثججججححححخخخخددذذررززسسسسششششصصصصضضضضططططظظظظععععغغغغففففققققککککللللممممننننههههوویییییییکی"" '
        )
        number_translation_src = "0123456789%٠١٢٣٤٥٦٧٨٩"
        number_translation_dst = "۰۱۲۳۴۵۶۷۸۹٪۰۱۲۳۴۵۶۷۸۹"
        suffixes = {
            "ی",
            "ای",
            "ها",
            "های",
            "هایی",
            "تر",
            "تری",
            "ترین",
            "گر",
            "گری",
            "ام",
            "ات",
            "اش",
        }
        extra_space_patterns = [
            (r" {2,}", " "),  # remove extra spaces
            (r"\n{3,}", "\n\n"),  # remove extra newlines
            (r"\u200c{2,}", "\u200c"),  # remove extra ZWNJs
            (r"\u200c{1,} ", " "),  # remove unneded ZWNJs before space
            (r" \u200c{1,}", " "),  # remove unneded ZWNJs after space
            (r"\b\u200c*\B", ""),  # remove unneded ZWNJs at the beginning of words
            (r"\B\u200c*\b", ""),  # remove unneded ZWNJs at the end of words
            (r"[ـ\r]", ""),  # remove keshide, carriage returns
        ]

        punc_after, punc_before = r"\.:!،؛؟»\]\)\}", r"«\[\(\{"

        punctuation_spacing_patterns = [
            # remove space before and after quotation
            ('" ([^\n"]+) "', r'"\1"'),
            (" ([" + punc_after + "])", r"\1"),  # remove space before
            ("([" + punc_before + "]) ", r"\1"),  # remove space after
            # put space after . and :
            (
                "([" + punc_after[:3] + "])([^ " + punc_after + r"\d۰۱۲۳۴۵۶۷۸۹])",
                r"\1 \2",
            ),
            (
                "([" + punc_after[3:] + "])([^ " + punc_after + "])",
                r"\1 \2",
            ),  # put space after
            (
                "([^ " + punc_before + "])([" + punc_before + "])",
                r"\1 \2",
            ),  # put space before
            # put space after number; e.g., به طول ۹متر -> به طول ۹ متر
            (r"(\d)([آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی])", r"\1 \2"),
            # put space after number; e.g., به طول۹ -> به طول ۹
            (r"([آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی])(\d)", r"\1 \2"),
        ]
        affix_spacing_patterns = [
            (r"([^ ]ه) ی ", r"\1‌ی "),  # fix ی space
            (r"(^| )(ن?می) ", r"\1\2‌"),  # put zwnj after می, نمی
            # put zwnj before تر, تری, ترین, گر, گری, ها, های
            (
                r"(?<=[^\n\d "
                + punc_after
                + punc_before
                + "]{2}) (تر(ین?)?|گری?|های?)(?=[ \n"
                + punc_after
                + punc_before
                + "]|$)",
                r"‌\1",
            ),
            # join ام, ایم, اش, اند, ای, اید, ات
            (
                r"([^ ]ه) (ا(م|یم|ش|ند|ی|ید|ت))(?=[ \n" + punc_after + "]|$)",
                r"\1‌\2",
            ),
            # شنبهها => شنبه‌ها
            ("(ه)(ها)", r"\1‌\2"),
        ]
        replacements = [
            ("﷽", "بسم الله الرحمن الرحیم"),
            ("﷼", "ریال"),
            ("(ﷰ|ﷹ)", "صلی"),
            ("ﷲ", "الله"),
            ("ﷳ", "اکبر"),
            ("ﷴ", "محمد"),
            ("ﷵ", "صلعم"),
            ("ﷶ", "رسول"),
            ("ﷷ", "علیه"),
            ("ﷸ", "وسلم"),
            ("ﻵ|ﻶ|ﻷ|ﻸ|ﻹ|ﻺ|ﻻ|ﻼ", "لا"),
        ]
        diacritics_patterns = [
            # remove FATHATAN, DAMMATAN, KASRATAN, FATHA, DAMMA, KASRA, SHADDA, SUKUN
            ("[\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652]", ""),
        ]
        specials_chars_patterns = [
            # Remove almoast all arabic unicode superscript and subscript characters in the ranges of 00600-06FF, 08A0-08FF, FB50-FDFF, and FE70-FEFF
            (
                "[\u0605\u0653\u0654\u0655\u0656\u0657\u0658\u0659\u065a\u065b\u065c\u065d\u065e\u065f\u0670\u0610\u0611\u0612\u0613\u0614\u0615\u0616\u0618\u0619\u061a\u061e\u06d4\u06d6\u06d7\u06d8\u06d9\u06da\u06db\u06dc\u06dd\u06de\u06df\u06e0\u06e1\u06e2\u06e3\u06e4\u06e5\u06e6\u06e7\u06e8\u06e9\u06ea\u06eb\u06ec\u06ed\u06fd\u06fe\u08ad\u08d4\u08d5\u08d6\u08d7\u08d8\u08d9\u08da\u08db\u08dc\u08dd\u08de\u08df\u08e0\u08e1\u08e2\u08e3\u08e4\u08e5\u08e6\u08e7\u08e8\u08e9\u08ea\u08eb\u08ec\u08ed\u08ee\u08ef\u08f0\u08f1\u08f2\u08f3\u08f4\u08f5\u08f6\u08f7\u08f8\u08f9\u08fa\u08fb\u08fc\u08fd\u08fe\u08ff\ufbb2\ufbb3\ufbb4\ufbb5\ufbb6\ufbb7\ufbb8\ufbb9\ufbba\ufbbb\ufbbc\ufbbd\ufbbe\ufbbf\ufbc0\ufbc1\ufc5e\ufc5f\ufc60\ufc61\ufc62\ufc63\ufcf2\ufcf3\ufcf4\ufd3e\ufd3f\ufe70\ufe71\ufe72\ufe76\ufe77\ufe78\ufe79\ufe7a\ufe7b\ufe7c\ufe7d\ufe7e\ufe7f\ufdfa\ufdfb]",
                "",
            ),
        ]


