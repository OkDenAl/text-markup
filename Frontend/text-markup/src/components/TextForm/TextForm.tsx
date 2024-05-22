import { FormEvent, useEffect, useRef } from "react";

import "./TextForm.scss";
import {
    useMarkup,
    useMarkupDispatch,
} from "../../contexts/MarkupProvider/MarkupHooks";
import { useNavigate } from "react-router-dom";

export interface TextFormProps {}

export default function TextForm() {
    const { text, loading, textClass } = useMarkup();
    const markupDispatch = useMarkupDispatch();
    const navigate = useNavigate();
    const loadingRef = useRef(false);

    useEffect(() => {
        if (loadingRef.current && !loading && textClass) {
            navigate("/markup");
        }
        loadingRef.current = loading;
    }, [loading, navigate, textClass]);

    function handleSubmit(e: FormEvent<HTMLFormElement>) {
        e.preventDefault();
        const textArea = e.currentTarget.elements.namedItem(
            "text"
        ) as HTMLTextAreaElement;
        if (!textArea.value) return;

        markupDispatch({ type: "FETCH_TEXT", payload: null });
    }

    return (
        <form className="text-form" onSubmit={handleSubmit}>
            <textarea
                className="text-form__textarea scrollable"
                name="text"
                id="text-form_text"
                placeholder="Вставьте текст..."
                value={text || ""}
                onChange={(e) =>
                    markupDispatch({
                        type: "TEXT",
                        payload: e.target.value,
                    })
                }
                disabled={loading}
            ></textarea>
            <button
                className="text-form__submit submit-button"
                type="submit"
                disabled={loading}
            >
                Получить разметку
            </button>
        </form>
    );
}
