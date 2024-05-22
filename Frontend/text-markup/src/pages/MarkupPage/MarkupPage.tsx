import MarkupViewer from "../../components/MarkupViewer/MarkupViewer";
import TextViewer from "../../components/TextViewer/TextViewer";

import Page from "../Page/Page";
import "./MarkupPage.scss";

export default function MarkupPage() {
    return (
        <Page>
            <div className="markup-page">
                <TextViewer />
                <MarkupViewer />
            </div>
        </Page>
    );
}
