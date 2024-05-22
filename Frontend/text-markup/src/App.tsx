import { RouterProvider, createBrowserRouter } from "react-router-dom";
import TextPage from "./pages/TextPage/TextPage";
import MarkupPage from "./pages/MarkupPage/MarkupPage";
import MarkupProvider from "./contexts/MarkupProvider/MarkupProvider";
import { Slide, ToastContainer } from "react-toastify";

import "./App.scss";
import "react-toastify/dist/ReactToastify.min.css";

const router = createBrowserRouter([
    {
        path: "/",
        element: <TextPage />,
    },
    {
        path: "/markup",
        element: <MarkupPage />,
    },
]);

export default function App() {
    return (
        <>
            <MarkupProvider>
                <RouterProvider router={router} />
            </MarkupProvider>
            <ToastContainer
                position="top-right"
                autoClose={1500}
                hideProgressBar
                newestOnTop
                closeOnClick
                rtl={false}
                pauseOnFocusLoss
                draggable
                pauseOnHover
                theme="light"
                transition={Slide}
            />
        </>
    );
}
