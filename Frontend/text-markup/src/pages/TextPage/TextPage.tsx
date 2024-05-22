import { useRef, PointerEvent } from "react";
import TextLoader from "../../components/TextLoader/TextLoader";
import Page from "../Page/Page";

import "./TextPage.scss";

export default function TextPage() {
    const textLoaderRef = useRef<HTMLDivElement>(null);
    const aboutRef = useRef<HTMLDivElement>(null);

    function handleTryItClick(e: PointerEvent<HTMLAnchorElement>) {
        e.preventDefault();

        textLoaderRef.current?.scrollIntoView({
            behavior: "smooth",
            block: "end",
        });
    }

    function handleAboutClick() {
        aboutRef.current?.scrollIntoView({
            behavior: "smooth",
            block: "center",
        });
    }

    return (
        <Page>
            <div className="text-page">
                <div className="text-page__text-loader" ref={textLoaderRef}>
                    <h2
                        className="text-page__text-loader__title"
                        onClick={handleAboutClick}
                    >
                        Вставьте текст или загрузите файл
                    </h2>
                    <TextLoader />
                </div>
                <div className="about" ref={aboutRef}>
                    <h2 className="about__header">О нас</h2>
                    <img
                        src="about.jpg"
                        alt="Разметка текста"
                        className="about__image"
                    />
                    <div className="about__text">
                        Добро пожаловать на наш сервис по получению тегов
                        текста, где точность сочетается с эффективностью.
                        Благодаря нашему удобному интерфейсу выделение ключевых
                        слов в тексте стало еще проще. Попрощайтесь с хаосом
                        неорганизованного контента и приветствуйте рациональный,
                        ясный и структурированный подход к определению и
                        выделению наиболее важных терминов. Наш
                        усовершенствованный алгоритм обеспечивает точность и
                        скорость, экономя ваше время и усилия. Присоединяйтесь к
                        нам и раскройте возможности эффективной разметки
                        ключевых слов для лучшего и более организованного
                        присутствия в Интернете.
                    </div>
                    <a
                        className="about__try-it"
                        href="#text-loader"
                        onClick={handleTryItClick}
                    >
                        Попробуйте сейчас
                    </a>
                </div>
                <div className="user-guide">
                    <h2 className="user-guide__header">Как это работает?</h2>
                    <div className="user-guide__text">
                        Чтобы получить теги из текста, просто вставьте его в
                        форму или же загрузите текст в виде файла. Наш сервис
                        определит класс текста - его главную тему, а также теги
                        - ключевые слова, связанные с темой. Помимо этого,
                        каждый тег будет иметь метку, которую можно удобно
                        использовать в дальнейшей программной обработке текста.
                    </div>
                    <img
                        src="user-guide.gif"
                        alt="Как это работает"
                        className="user-guide__image"
                    />
                </div>
            </div>
        </Page>
    );
}
