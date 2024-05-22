import { ChangeEvent, FormEvent, useEffect, useRef, useState } from "react";
import { toast } from "react-toastify";
import {
    useMarkup,
    useMarkupDispatch,
} from "../../contexts/MarkupProvider/MarkupHooks";
import FileDropArea from "../FileDropArea/FileDropArea";

import "./FileForm.scss";
import { useNavigate } from "react-router-dom";

export interface FileFormProps {}

export default function FileForm() {
    const [file, setFile] = useState<File | null>(null);
    const { loading, textClass } = useMarkup();
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
        if (!file) {
            toast.error("Выберите файл");
            return;
        }

        markupDispatch({ type: "FETCH_FILE", payload: file });
    }

    function handleFileLoad(e: ChangeEvent<HTMLInputElement>) {
        setFile(e.currentTarget.files!.item(0));
    }

    return (
        <>
            <form className="file-form" onSubmit={handleSubmit}>
                <FileDropArea
                    className="file-form__file-area"
                    onFile={(file) => setFile(file)}
                >
                    <input
                        type="file"
                        name="file"
                        className="file-form__file-area__input"
                        id="file-form__file"
                        onChange={handleFileLoad}
                        disabled={loading}
                        hidden
                        accept=".doc,.docx,.txt"
                    />
                    <label
                        className="file-form__file-area__label"
                        htmlFor="file-form__file"
                    >
                        {file ? file.name : "Выберите файл или перетащите его"}
                    </label>
                    <span className="file-form__file-area__accept-files">
                        Разрешенные форматы .doc, .docx, .txt
                    </span>
                </FileDropArea>
                <button
                    className="file-form__submit submit-button"
                    type="submit"
                    disabled={loading}
                >
                    Получить разметку
                </button>
            </form>
        </>
    );
}
