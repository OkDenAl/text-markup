import { ReactNode } from "react";
import "./Page.scss";
import { Link } from "react-router-dom";

export interface PageProps {
    children: ReactNode;
}

export default function Page({ children }: PageProps) {
    return (
        <div className="page-container">
            <header className="header">
                <Link className="header__title" to="/">
                    Cluster text
                </Link>
            </header>
            <main className="main">{children}</main>
        </div>
    );
}
