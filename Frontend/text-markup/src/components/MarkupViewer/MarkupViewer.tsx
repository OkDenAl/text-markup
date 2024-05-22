import { useMarkup } from "../../contexts/MarkupProvider/MarkupHooks";
import "./MarkupViewer.scss";
export interface MarkupViewerProps {}

export default function MarkupViewer() {
    const { tags, labels, textClass, keywords } = useMarkup();

    const items = [];
    for (let i = 0; i < tags.length; i++) {
        items.push({
            tag: tags[i],
            label: labels[i],
        });
    }

    return (
        <div className="markup-viewer scrollable">
            <div className="markup-viewer__text-class">
                <h3 className="markup-viewer__text-class__title">
                    Класс текста:
                </h3>
                <span className="markup-viewer__text-class__class-value">
                    {textClass}
                </span>
            </div>
            <div className="markup-viewer__keywords">
                <h3 className="markup-viewer__keywords__title">
                    Ключевые слова:
                </h3>
                <div className="markup-viewer__keywords__words">
                    {keywords.map((keyword) => (
                        <span className="markup-viewer__keywords__words__word">
                            {keyword}
                        </span>
                    ))}
                </div>
            </div>
            <div className="markup-viewer__tags-labels">
                <h3 className="markup-viewer__tags-labels__tags-title">Теги</h3>
                <h3 className="markup-viewer__tags-labels__labels-title">
                    Метки
                </h3>
                {items.map(({ tag, label }) => (
                    <>
                        <div
                            key={tag}
                            className="markup-viewer__tags-labels__tag"
                        >
                            {tag}
                        </div>
                        <div
                            key={label}
                            className="markup-viewer__tags-labels__label"
                        >
                            {label}
                        </div>
                    </>
                ))}
            </div>
        </div>
    );
}
