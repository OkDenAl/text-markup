import { useState } from "react";
import Chooser from "../Chooser/Chooser";
import FileForm from "../FileForm/FileForm";
import TextForm from "../TextForm/TextForm";

import "./TextLoader.scss";

export type TextSource = "Текст" | "Файл";

export interface TextLoaderProps {}

export default function TextLoader() {
    const textSourses: TextSource[] = ["Текст", "Файл"];
    const [textSource, setTextSource] = useState<TextSource>("Текст");

    return (
        <div className="text-loader">
            <Chooser
                options={textSourses}
                startOption={textSource}
                onChoose={(textSrc) => setTextSource(textSrc as TextSource)}
            />
            {textSource == "Текст" && <TextForm />}
            {textSource == "Файл" && <FileForm />}
        </div>
    );
}
