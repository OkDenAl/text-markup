from embedder.emb import Embedder

from torch.nn.functional import cosine_similarity

if __name__ == "__main__":
    embdr = Embedder('DeepPavlov/rubert-base-cased-sentence', 'cpu')
    text1 = 'Москва. 15 марта. INTERFAX.RU - В регионах РФ начинается трехдневное голосование на выборах президента - оно пройдет с 15 по 17 марта; при этом первые избирательные участки открылись в Камчатском крае в 23:00 14 марта (по московскому времени).\
    \
    Центризбирком России зарегистрировал кандидатами на выборах главу партии ЛДПР Леонида Слуцкого, вице-спикера Госдумы, члена партии "Новые люди" Владислава Даванкова, депутата Госдумы (фракция КПРФ) Николая Харитонова и действующего главу государства, самовыдвиженца Владимира Путина.\
    \
    Электронное голосование\
    Механизм дистанционного электронного голосования впервые будет применяться на выборах президента. Оно пройдет в 29 регионах России: в 28 регионах - на федеральной платформе, в Москве - на собственной региональной платформе. Прием документов для участия в онлайн-голосовании начался 29 января и завершился 11 марта.\
    \
    Ранее глава ЦИК РФ Элла Памфилова сообщила, что для участия в ДЭГ подано около 4 млн 787 тыс. заявлений.\
    \
    Подсчитывать итоги онлайн-голосования планируется в два этапа. Во всех субъектах страны, кроме Калининградской области, выборы завершатся в 20:00 17 марта, после этого начнется подведение итогов. Первые результаты электронного голосования ЦИК планирует сообщить после 21:00 (по Москве) 17 марта, после чего начнется подсчет данных в Калининградской области.\
    \
    Проголосовать онлайн на выборах избиратели смогут на интернет-портале vybory.gov.ru (в Москве - на портале mos.ru) с 08:00 15 марта до 20:00 17 марта по местному времени.\
    \
    "Мобильный избиратель"\
    Механизм "Мобильный избиратель" позволяет проголосовать на любом удобном для гражданина избирательном участке в России, а не только по месту постоянной или временной регистрации.\
    \
    Как ранее сообщила Памфилова, для голосования по месту нахождения на выборах через этот механизм подано более 5 млн 195 тыс. заявлений.'

    text2 = 'Первые избирательные участки открылись ровно в 23:00 мск. Впервые в истории страны основное голосование проходит в течение трех дней – 15, 16 и 17 \
    Как пишут российские СМИ, свои двери по всей стране откроют свыше 94 тысяч избирательных участков, численность избирателей – 112,3 млн человек в России и 1,9 млн за рубежом.\
    Впервые для выборов президента России в ряде регионов используется дистанционное электронное голосование.\
    На пост главы российского государства претендуют четыре кандидата: Владислав Даванков, Владимир Путин, Леонид Слуцкий и Николай Харитонов.\
    Россияне, которые находятся за рубежом, смогут проголосовать на выборах на 295 избирательных участках в 144 странах, сообщили в ЦИК.\
    Последние избирательные участки закроются в воскресенье, 17 марта, в 21:00 мск в Калининградской области.\
    Следить за выборами будут более 156 тысяч общественных наблюдателей, есть и иностранные из 106 государств.\
    Нынешние президентские выборы – восьмые в истории России.\
    Ранее Совет Федерации назначил выборы президента России на 17 марта 2024 года. Центризбирком РФ решил, что голосование продлится три дня: 15, 16 и 17 марта. Таким образом, это будут первые трехдневные выборы президента РФ.\
    1 января 2024 года стало известно, что Леонид Слуцкий и Сергей Малинкович подали документы в Центральную избирательную комиссию РФ для регистрации своих кандидатур на выборах президента России.'

    text3 = 'Рекс Ремингтон, который уже 40 лет занимается рыбалкой, рассказал о своем улове, но решил сохранить в тайне некоторые подробности.\
    В США 60-летний Рекс Ремингтон из штата Индиана поймал крупного малоротого окуня и побил рекорд, который держался с 1992 года. Об этом пишет Reporter-Times.\
    Во время рыбалки на озере Монро в недавнее воскресенье, 10 марта, мужчина вытащил большую рыбу, вес которой составил 8,23 фунта (3,7 кг). Предыдущий рекорд был поставлен в округе ЛаГрейндж в 1992 году, он составлял чуть более 7 фунтов (3,3 кг).\
    60-летний американец рассказал, что рыбачит на озере Монро всю жизнь.\
    "Я помню, как ходил на озеро еще в детстве. Мама и папа много лет ловили тут рыбу, а мне сейчас 60. Сколько я целенаправленно охочусь за окунями? Лет 30-40", — рассказал мужчина.'

    text4 = 'Вратарь и капитан московского ЦСКА Игорь Акинфеев рассказал, что изменил бы в российском футболе.\
    "Раньше академии работали замечательно, сборная России была на шестом месте в рейтинге ФИФА. Ушло старое поколение — произошёл провал. Видимо, надо изменения с академий начинать. Но это заезженная тема. Я был маленький, и все говорили, что надо с академий начинать. Сейчас мне скоро уже 50 лет, и теперь я рассказываю, что надо с академиями что-то делать. Но это правда, это касается всех видов спорта. Приезжайте в регионы, посмотрите, что там творится.\
    Если бы у меня была возможность что-то изменить, я бы начал с академий. Всё полностью надо менять: от полей и раздевалок до тренеров. Человеку нужен комфорт. Когда мы были в Абу-Даби, я поразился количеству детских площадок; тысячи полей, бесплатных секций. У нас такого нет — это одна из главных проблем«, — сказал Акинфеев корреспонденту “Чемпионата”.\
    Игорь Акинфеев дебютировал за ЦСКА в 2003 году. В активе голкипера 560 матчей в РПЛ и 248 матчей «на ноль». Оба показателя являются рекордными в чемпионатах России. Акинфеев сыграл 111 матчей за сборную России. После домашнего чемпионата мира 2018 года он объявил об уходе из национальной команды.'

    text5 = 'Бывший президент США Дональд Трамп и нынешний хозяин Белого дома Джо Байден победили на праймериз республиканцев и демократов, сообщают американские СМИ. Таким образом они вскоре будут официально выдвинуты в президенты США от своих партий — и выборы 2024 года станут фактическим повторением кампании 2020 года, когда Байден победил Трампа.\
    Выдвижение Байдена в президенты США от Демократической партии было формальностью, поскольку действующий лидер страны, решивший идти на второй срок, как правило, не сталкивается с серьезной конкуренцией.\
    У республиканцев битва за выдвижение была вполне серьезной, хотя Трамп с самого начала кампании имел огромное преимущество и у его ближайшей конкурентки, бывшей постоянной представительницы США в ООН Никки Хейли, почти не было реальных шансов на выдвижение.\
    Во вторник республиканские праймериз прошли в четырех штатах: в Джорджии, на Гавайях, а также в Миссисипи и Вашингтоне.\
    После того, как Хейли вышла из гонки на фоне поражений в «супервторник», Трампу не хватало всего 137 делегатов на партийной конференции, которая вскоре официально выдвинет кандидата в президенты.\
    Единственная соперница Трампа Никки Хейли вышла из гонки за номинацию от республиканцев, но не поддержала экс-президента\
    Как передает ABC News, экс-президент получил необходимые голоса после победы в Вашингтоне.'

    text6 = 'Действующий обладатель Кубка России московский ЦСКА в полуфинале Пути РПЛ сыграет с калининградской «Балтикой». В другой паре жребий свел московский «Спартак» и петербургский «Зенит».\
    Пары ½ финала Пути РПЛ Кубка России\
    «Спартак» — «Зенит»\
    «Балтика» — ЦСКА\
    Первые полуфинальные матчи Пути РПЛ Кубка России пройдут 2—4 апреля, ответные — 16—18 апреля. Команды, указанные первыми, сыграют первые матчи дома.\
    «Оренбург», «Локомотив», «Динамо» и «Ростов», вылетевшие из Пути РПЛ на стадии четвертьфинала, продолжают борьбу за трофей в Пути регионов.\
    Пары второго этапа ¼ финала Пути регионов Кубка России\
    «Ахмат» — «Оренбург»\
    «Урал» — «Локомотив»\
    «СКА-Хабаровск» — «Динамо»\
    «Химки» — «Ростов»\
    Команды, указанные первыми, будут играть дома. На всех стадиях Пути регионов команды проводят по одной игре. Матчи второго этапа ¼ финала Пути регионов пройдут 2—4 апреля.'
    embed1 = embdr.get_embedding(text1)
    embed2 = embdr.get_embedding(text2)
    embed3 = embdr.get_embedding(text3)
    embed4 = embdr.get_embedding(text4)
    embed5 = embdr.get_embedding(text5)
    embed6 = embdr.get_embedding(text6)

    elec_vs_elec = cosine_similarity(embed1, embed2)
    elec_vs_fishing = cosine_similarity(embed1, embed3)
    elec_vs_fishing2 = cosine_similarity(embed2, embed3)
    elec_vs_football = cosine_similarity(embed1, embed4)
    fishing_vs_football = cosine_similarity(embed3, embed4)
    elec_vs_US_elec = cosine_similarity(embed1, embed5)
    footbal_vs_football = cosine_similarity(embed4, embed6)
    print(elec_vs_elec, elec_vs_fishing, elec_vs_fishing2, elec_vs_football, fishing_vs_football, elec_vs_US_elec, footbal_vs_football)