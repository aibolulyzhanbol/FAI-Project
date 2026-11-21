# Дереккөздер және деректердің шығу тегі / Data provenance

Барлық дереккөздер 14.11.2026 күні ашылды. Бұл жарияланымдардағы SMS мәтіндері датасетке сөзбе-сөз көшірілмеген:
әр хабарлама — дереккөзде сипатталған схеманың қысқа реконструкциясы (`text_provenance` бағаны).

| Кілт | Дереккөз | Не пайдаланылды |
|---|---|---|
| KZ_POST | https://newtimes.kz/obshchestvo/225922-utochnite-adres-dostavki-kazakhstantsam-rassylaiut-opasnye-sms-ot-kazpochty | «жеткізу мекенжайын нақтылаңыз» SMS (Бас прокуратура, Қазпошта, 11.11.2026), жалған курьерлер, Beeline/Halyk атынан жіберілген хабарламалар, iPhone 18 алдын ала тапсырысы |
| HALYK | https://finratings.kz/news/13107-kazakhstantsev-atakuiut-moshenniki-cherez-sms-ot-halyk-bank/ | «Halyk Bank атынан» бонустар туралы SMS |
| KASPI_WA | https://finratings.kz/news/1265-kaspi-kz-moshenniki-rassylaiut-falshivye-ssylki-v-whatsapp/ | бұзылған аккаунттардан жіберілетін сілтемелер; реквизиттер мен кодтарды айтпау туралы кеңес |
| EGOV_FINE | https://tengrinews.kz/kazakhstan_news/kazahstantsyi-riskuyut-poteryat-dengi-uplate-shtrafov-596679/ | жалған eGov жалған айыппұл көрсетеді |
| EGOV_FINE2 | https://www.lada.kz/kazakhstan-news/151642-shtrafy-na-egov-opustoshaiut-bankovskie-karty-kazakhstantsev-chto-proiskhodit.html | eGov айыппұлдары, 1414 нөмірі, eGov Mobile / qamqor.gov.kz арқылы тексеру |
| EGOV_STAFF | https://finratings.kz/news/5868-ostorozhno-moshenniki-vazhnoe-preduprezhdenie-dlia-kazakhstantsev/ | «eGov.kz қызметкерлері» |
| EGOV_SEMEY | https://www.zakon.kz/obshchestvo/6524297-moshenniki-stanovyatsya-pravdopodobnee-vazhnoe-preduprezhdenie-sdelali-kazakhstantsam.html | ИИН/CVV/құпиясөздер/кодтарды айтпау туралы ескерту |
| SMISHING | https://finance.kz/articles/smishing-sms-moshenniki-kak-zashchititsya | айыппұл, несие, SIM-карта, «қауіпсіз шот», 1414-тен келген SMS |
| PHISHING_KZ | https://finance.kz/articles/phishing | Қазақстандағы фишинг, KZ-CERT ұсынымдары |
| QAZPOST | https://finratings.kz/news/1206-fishingovaia-ataka-v-kazakhstane-moshenniki-maskiruiutsia-pod-qazpost/ | Qazpost атынан жіберілген SMS |
| ADCB | https://www.adcb.com/ru/kazakhstan/business/security-tips/fraud-prevention/ | банк ескертуі (1414, eGov, QazPost, E-Salyk) |

## Бағандардың мәндері
- `text_provenance`: `documented_scheme_paraphrase` (схема дереккөзде сипатталған), `official_advisory_paraphrase`
  (ресми ескертудің мазмұны — бұл «қорқынышты» сөздері бар SAFE мәтін, классификатор үшін әдейі қиын), `synthetic_realistic`
  (жоба авторы құрастырған, сыртқы дереккөз жоқ).
- `url_provenance`: `official_domain` (нақты ресми домен), `constructed_lookalike` (сілтеме «бренд + арзан TLD» үлгісімен ойдан шығарылған;
  бұл НАҚТЫ фишинг сілтемесі емес, оны ашпаңыз және тіркемеңіз), `none` (сілтеме жоқ).
- `group`: алаяқтық схемасы. Train/Test топтар бойынша бөлінген, сондықтан бір схеманың парафраздары екі бөлікке де түспейді.

## Нақты деректер ЕМЕС нәрселер (адал айтамыз)
Тірі фишинг URL-дері мен сөзбе-сөз SMS-тер ашық мақалаларда дерлік жарияланбайды. Сондықтан датасет — құжатталған схемалардың
реконструкциясы, нақты трафиктен алынған жүктеме емес. Нақты URL үшін `tools/import_openphish.py` қараңыз.
