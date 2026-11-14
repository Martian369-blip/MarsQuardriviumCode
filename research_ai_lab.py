import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import sympy as sp


def solve_symbolic_math(expression_str, mode="Simplify / Evaluate", variable_str="x"):
    """Uses SymPy to solve equations, differentiate, integrate, or simplify symbolic expressions."""
    try:
        var = sp.Symbol(variable_str.strip() or "x")
        if "=" in expression_str and mode == "Solve Equation (= 0)":
            lhs_str, rhs_str = expression_str.split("=", 1)
            expr = sp.sympify(lhs_str) - sp.sympify(rhs_str)
        else:
            expr = sp.sympify(expression_str)

        if mode == "Solve Equation (= 0)":
            result = sp.solve(expr, var)
        elif mode == "Differentiate (d/dx)":
            result = sp.diff(expr, var)
        elif mode == "Integrate (∫ dx)":
            result = sp.integrate(expr, var)
        elif mode == "Series Expansion (Taylor)":
            result = sp.series(expr, var, 0, 6).removeO()
        else:
            result = sp.simplify(expr)

        return {
            "status": "success",
            "input_latex": sp.latex(expr),
            "result_str": str(result),
            "result_latex": sp.latex(result),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def query_arxiv_papers(query, max_results=5):
    """Queries the open-access arXiv API for preprints in physics, astronomy, mathematics, and computing."""
    encoded_q = urllib.parse.quote(query)
    url = f"http://arxiv.org:{encoded_q}&start=0&max_results={max_results}"
    papers = []
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            ns = {"atom": "http://w3.org"}
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
                summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
                link = entry.find("atom:id", ns).text.strip()
                published = entry.find("atom:published", ns).text[:10]
                authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
                papers.append({
                    "Title": title,
                    "Authors": ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""),
                    "Published": published,
                    "Summary": summary[:320] + "...",
                    "URL": link,
                })
    except Exception as e:
        return [{"Error": f"arXiv query failed: {e}"}]
    return papers


def query_crossref_research(query, domain_filter="", max_results=5):
    """Queries Crossref academic metadata API for peer-reviewed papers in Anthropology, Archaeology, Astronomy, and Science."""
    full_query = f"{domain_filter} {query}".strip()
    encoded_q = urllib.parse.quote(full_query)
    url = f"https://crossref.org{encoded_q}&rows={max_results}"
    articles = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "QuadriviumResearchEngine/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            items = data.get("message", {}).get("items", [])
            for item in items:
                title_list = item.get("title", ["Untitled"])
                title = title_list[0] if title_list else "Untitled"
                doi = item.get("DOI", "")
                pub_year = item.get("published", {}).get("date-parts", [[None]])[0][0]
                journal = item.get("container-title", ["Academic Journal"])
                journal_name = journal[0] if journal else "Academic Journal"
                articles.append({
                    "Title": title,
                    "Journal / Source": journal_name,
                    "Year": str(pub_year) if pub_year else "N/A",
                    "DOI Link": f"https://doi.org{doi}" if doi else "N/A",
                })
    except Exception as e:
        return [{"Error": f"Crossref query failed: {e}"}]
    return articles


def query_encyclopedic_synthesis(topic):
    """Fetches structured scientific and historical summary data from Wikipedia REST API."""
    clean_topic = urllib.parse.quote(topic.strip().replace(" ", "_"))
    url = f"https://wikipedia.org{clean_topic}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "QuadriviumResearchEngine/1.0"})
        with urllib.request.urlopen(req, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))
            return {
                "title": data.get("title", topic),
                "description": data.get("description", ""),
                "extract": data.get("extract", "No summary available."),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            }
    except Exception:
        return None
