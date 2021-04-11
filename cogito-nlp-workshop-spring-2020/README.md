# Workshop med Strise
I denne workshopen skal vi benytte oss av et datasett som inneholder anmeldelser av kvinneklær på nett.
Oppgavene vil i stor grad kunne bestemmes av dere selv, men vi har også kommet med noen forslag.

## Setup
Klon eller last ned dette repoet. Dette gjøres ved å klikke på den store grønne knappen oppe til høyre, eller ved å kjøre `git clone https://github.com/strise/cogito-workshop-spring-2020`. 
Du trenger minimum Python 3.6 for å kjøre koden. Dersom du ikke har det kan du laste det ned [her](https://www.python.org/downloads/). 

Naviger deretter til mappen med repoet i terminalen og kjør kommandoen `pip3 install -r requirements.txt` for å installere avhengigheter. Du må også laste ned et par såkalte "corpus" til nltk. Dette gjøres ved å kjøre `CorpusDownloader.py`, som ligger i `utils`-mappen. 

Hvis du står fast her, ikke nøl med å spørre!


## Oppgaver
I denne workshopen er det ingen krav, men målet er heller at dere skal få leke litt med datasettet som vi har klargjort.
For å gi litt inspirasjon har vi inkludert noen eksempler som dere kan bruke. Vi har også skrevet noen forslag til 
oppgaver, dersom dere sliter med kreativiteten. Det anbefales også å se nøye på hvilken data som er inkludert i datasettet:
Er det noe som kunne ha vært spennende å se på?

### Forslag til oppgaver
For å komme raskt i gang har vi laget noen forslag til oppgaver. Disse kan sees på som inspirasjon, så finn gjerne på andre vinklinger som du finner mer interessante:
- List ut de 10 mest brukte adjektivene for henholdsvis personer under 25, og personer over 60. Er det noe forskjell?
- List ut de 10 mest brukte adjektivene for de 5 forskjellige verdiene av `rating`.
- Er det en sammenheng mellom `upvotes` og `rating` fra anmeldelse? (se `SentimentExample.py` for inspirasjon til 
plotting av slik data)
- Er det forskjell i hvor positive anmeldelsene for de forskjellige kategoriene er? For eksempel, er anmeldelser på bukser
mer eller mindre positive enn anmeldelser på svømmetøy? Her er det både mulig å se på `rating`, gitt fra bruker, og å 
finne sentiment for anmeldelsene (se `SentimentExample.py`)

Om du vil ha noe som er litt vanskeligere:
- Lag et program som predikerer `rating` på anmeldelsene. Her er det flere muligheter:
    - Lag en regelbasert implementasjon som benytter sentiment (se `SentimentExample.py`) og eventuelt andre måltall.
    - Mer avansert: En maskinlæringsmodell som tar inn de forskjellige verdiene i en anmeldelse. Her lønner det seg å ha
    jobbet litt med slike metoder tidligere. 
- Hva er det som gjør at et review får mange upvotes? Finner du noen mønster?
- Basert på anmeldelsestekstene, lag en "chatbot". Som i `VectorizerExample.py` kan du transformere hver setning i 
datasettet til en vektor. Når bruker skriver til chatboten, finn den setningen som ligner mest på brukerens input. Denne
chatboten kommer nok ikke til å klare [Turingtesten](https://no.wikipedia.org/wiki/Turingtest), men så lenge du snakker om
klær burde den klare å svare med noe som er innenfor samme tema.

## Dokumentasjon
### Datsett/Review-klassen
Datasettet er lastet ned fra [kaggle](https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews), og inneholder
data fra ekte anmeldelser av kvinneklær. For at dere skal slippe å bruke for mye tid på å laste inn datasett har vi skrevet
kode for dette. Ved å kjøre `DatasetLoader.load_reviews()` får man opprettet i overkant av 23.000 instanser av `Review`-klassen.
Den inneholder følgende felter:
- `review_id: str`: En unik ID for anmeldelsen.
- `product_id str`: IDen til produktet som omtales i anmeldelsen.
- `age: int`: Alderen til anmelderen.
- `title: str`: Tittel på anmeldelsen.
- `review_text: str`: Teksten i anmeldelsen.
- `rating: int`: Et score gitt fra bruker i rekkevidde [1, 5] (hvor 5 er best). 
- `recommends: bool`: En boolsk variabel som sier om brukeren anbefaler produktet eller ikke.
- `upvotes: int`: Antall brukere som har markert anmeldelsen som god.
- `division_name: str`: Høyeste kategori, som er en av følgende: `["General", "General Petite", "Intimates"]`
- `department_name: str`: Kategori for produktavdeling, som er en av følgende: `["Intimate", "Dresses", "Bottoms", "Tops", "Jackets", "Trend"]`
- `class_name: str`: Kategori for produkt, som er en av følgende: `["Intimates", "Dresses", "Pants", "Blouses", "Knits", "Outerwear", "Lounge", "Sweaters", "Skirts", "Fine gauge", "Sleep", "Jackets", "Swim", "Trend", "Jeans", "Legwear", "Shorts", "Layering", "Casual bottoms", "Chemises" ]`

Merk at noen av radene i datasettet kan ha tomme verdier på noen felter.

I tillegg har vi laget noen forhåndsdefinerte funksjoner i `Review`-klassen, som kan være nyttig å bruke:
- `full_text() -> str`: Returnerer `title` og `review_text` slått sammen til én streng.
- `words() -> List[str]`: Returnerer en liste med ordene i `full_text()`, vha. biblioteket nltk. 
- `tagged_words() -> List[Tuple[str, str]]`: Returnerer en liste med tupler for alle ordene i `words()`. Det første 
elementet i tuplene er order, mens det andre elementet beskriver hvilken grammatisk klasse ordet tilhører. 

`tagged_words()` tar noe tid å kjøre (og dette gjelder også noen andre funksjoner som benyttes i eksemplene). Dersom du opplever at det tar lang tid å kjøre koden din kan det være lurt å midlertidig bruke `DatasetLoader.load_reviews()[:100]` for å kun hente de første 100 anmeldelsene. 

## Eksempler
I `examples`-mappen ligger det noen eksempler på hva man kan gjøre med datasettet. Her kan man hente litt inspirasjon dersom
man står fast.

### `SentimentExample.py`
[Sentimentanalyse](https://en.wikipedia.org/wiki/Sentiment_analysis) er et felt innenfor 
naturlig språk-prosessering hvor man prøver å si om en tekst er positiv eller negativ. Det er ganske avansert, men i dette
eksempelet brukes et bibliotek som heter [TextBlob](https://textblob.readthedocs.io/en/dev/). Her dytter man inn en tekst,
og får ut et tall mellom -1 og 1 som sier noe om hvor positiv teksten er. TextBlob har ikke en veldig nøyaktig modell, 
men når man tar et gjennomsnitt av mange eksempler får man ofte ut noe fornuftig. 

I dette eksemplet hentes gjennomsnittlig sentiment for hver av de fem verdiene for `rating`, og plottes. Her kan man
se en tydelig sammenheng mellom sentiment og `rating`.

### `WordcloudExample.py`
I dette eksemplet lages en ordsky av alle verb i alle anmeldelser. Ved å kalle `tagged_words()`-funksjonen på `Review`-instanser får man tilbake tupler med
ord og grammatiske klasser ([liste over grammatiske klasser](https://pythonprogramming.net/natural-language-toolkit-nltk-part-speech-tagging/)).
Ved å kun ta med ord som har grammatisk klasse som begynner med `"VB"` får vi med verb i forskjellige bøyningsformer.

Her er det selvsagt enkelt å kun bruke adjektiver (`"JJ"`) eller substantiver (`"NN"`).

### `VectorizerExample.py`
I dette eksemplet presenteres et program hvor bruker må skrive en anmeldelse, og får servert den "likeste" anmeldelsen tilbake.
Her benyttes en metode som heter [Bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model). I korte trekk oppretter
man en vektor hvor hver indeks samsvarer med et ord blant anmeldelsene. For eksempel kan indeks 159 referere til ordet
"kjole". Når man transformerer en tekst til en slik vektor vil verdien i hver indeks fortelle hvor mange ganger et bestemt
ord er nevnt. Dette er en nyttig representasjon av tekst: Maskinlæringsalgoritmer kan ikke ta ren tekst som input, og er 
avhengig av å kunne gjøre om tekst til tall. 

Når eksemplet kjøres blir man spurt om å skrive en anmeldelse som blir transformert til en vektor.
Til slutt finner vi anmeldelsen i datasettet som har likest vektor, og printer denne.

Skriptet har ikke mye praktisk verdi, men eksemplet viser hvordan tekst kan gjøres om til tall. Dersom du planlegger å 
dytte anmeldelsene inn i et nevralt nettverk kan dette være en strategi å bruke.



# English: Workshop with Strise
In this workshop, we will use a dataset consisting of reviews of women's clothes. You can pretty much do whatever you want, but we have made some suggestions.

## Setup
Clone or download this repo (`git clone https://github.com/strise/cogito-workshop-spring-2020`). You'll need Python 3.6 (or newer) to run the code.

Navigate to the folder with the repo, and run `pip3 install -r requirements.txt` to install dependencies. You also need to download some corpuses for nltk. This can be done by running `CorpusDownloader.py`, which is located in the `utils`-folder.

If you get stuck, don't hesistate to ask for help!


## Tasks
There are no requirements for this workshop. The goal is rather to have fun and play with the dataset we have prepared. To give some inspiration, we have included some examples you can use. We have also written some suggestions for tasks, if you are not in the creative corner today. 

### Suggestions
The following tasks should be viewed as inspiration, and you are free to come up with your own angels on these tasks.
- List the 10 most used adjectives for persons under 25, and persons over 60, respectively.
- List the 10 most used adjectives for the five different values of `rating`.
- Is there any correlation between `upvotes` and `rating`? (See `SentimentExample.py` for inspiration on how to plot such data)
- Are there any differences between the sentiment of the reviews for the different categories of clothes? E.g., are the reviews for pants more positive than reviews for sweaters? (see `SentimentExample.py`)

If you want something more advanced:
- Create a program that predicts `rating` for a review. Here, there are many opportunities:
    - Create a rule based implementation which uses sentiment, and possibly other metrics.
    - More advanced: A machine learning model taking as input the different values in a review.
- Are there any patterns in the reviews receiving many upvotes?
- Based on the review texts, create a "chatbot". Like in `VectorizerExample.py`, you can transform each sentence in the dataset to a vector. When a user writes to the chatbot, find the sentence most similar to the user's input. This chatbot is probably not going to pass the [Turingtesten](https://no.wikipedia.org/wiki/Turingtest), but as long as you keep conversation about clothes, you should get responses that are similar to questions.
 
## Documentation
### Datset/The Review-class
The dataset is downloaded from [kaggle](https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews), and contains data from real reviews of women's clothes. We have created methods for parsing the dataset in `DatasetLoader.load_reviews()`. By running this function, you should get about 23.000 instances of the `Review`-class.

It contains the following fields:
- `review_id: str`: A unique ID for the review.
- `product_id str`: The ID of the reviewed product.
- `age: int`: The age of the reviewer.
- `title: str`: The title of the review.
- `review_text: str`: The body text of the review.
- `rating: int`: A score in the range [1,5] (where 5 is best)
- `recommends: bool`: En boolean variable stating whether this product was recommended by the reviewer or not.
- `upvotes: int`: Number of users that have marked the review as good.
- `division_name: str`: The highest level product category, which is one of the following: `["General", "General Petite", "Intimates"]`
- `department_name: str`: Category for department, which is one of the following: `["Intimate", "Dresses", "Bottoms", "Tops", "Jackets", "Trend"]`
- `class_name: str`: Product class, which is one of the following: `["Intimates", "Dresses", "Pants", "Blouses", "Knits", "Outerwear", "Lounge", "Sweaters", "Skirts", "Fine gauge", "Sleep", "Jackets", "Swim", "Trend", "Jeans", "Legwear", "Shorts", "Layering", "Casual bottoms", "Chemises" ]`

Note that some of the values might be empty.

Additionally, we have made some predefined functions which can be called on `Review`-instances:
- `full_text() -> str`: Returns `title` and `review_text` joined to one string.
- `words() -> List[str]`: Returns a list of the words in `full_text()`. 
- `tagged_words() -> List[Tuple[str, str]]`: Returns a list of tuples, based on the elements in `words()`. The first element in the tuples is the word, while the second element describes which grammatical category the word belongs to.  

## Examples
In the `examples`-folder, there are some examples on what you can do with the data set.
