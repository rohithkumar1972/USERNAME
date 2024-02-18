import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


# def transition_model(corpus, page, damping_factor):
#     """
#     Return a probability distribution over which page to visit next,
#     given a current page.

#     With probability `damping_factor`, choose a link at random
#     linked to by `page`. With probability `1 - damping_factor`, choose
#     a link at random chosen from all pages in the corpus.
#     """
#     raise NotImplementedError


# def sample_pagerank(corpus, damping_factor, n):
#     """
#     Return PageRank values for each page by sampling `n` pages
#     according to transition model, starting with a page at random.

#     Return a dictionary where keys are page names, and values are
#     their estimated PageRank value (a value between 0 and 1). All
#     PageRank values should sum to 1.
#     """
#     raise NotImplementedError


# def iterate_pagerank(corpus, damping_factor):
#     """
#     Return PageRank values for each page by iteratively updating
#     PageRank values until convergence.

#     Return a dictionary where keys are page names, and values are
#     their estimated PageRank value (a value between 0 and 1). All
#     PageRank values should sum to 1.
#     """
#     raise NotImplementedError

def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    num_pages = len(corpus)
    transition_probabilities = {}

    if len(corpus[page]) == 0:
        # If the current page has no outgoing links, distribute the probability equally among all pages
        return {p: 1 / num_pages for p in corpus}

    for p in corpus:
        # With probability damping_factor, choose a link at random linked to by `page`
        link_probability = damping_factor * (1 / len(corpus[page])) if p in corpus[page] else 0

        # With probability 1 - damping_factor, choose a link at random chosen from all pages in the corpus
        random_link_probability = (1 - damping_factor) / num_pages

        # Calculate the overall probability for the page
        transition_probabilities[p] = link_probability + random_link_probability

    return transition_probabilities


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_rank = {page: 0 for page in corpus}
    current_page = random.choice(list(corpus.keys()))

    for _ in range(n):
        # Update the PageRank values based on the current page
        page_rank[current_page] += 1
        transition_probabilities = transition_model(corpus, current_page, damping_factor)
        current_page = random.choices(list(transition_probabilities.keys()), weights=list(transition_probabilities.values()))[0]

    # Normalize PageRank values
    page_rank = {page: value / n for page, value in page_rank.items()}

    return page_rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    num_pages = len(corpus)
    page_rank = {page: 1 / num_pages for page in corpus}
    new_page_rank = page_rank.copy()

    while True:
        max_change = 0

        for page in corpus:
            # Calculate the new PageRank value for each page based on the PageRank formula
            new_page_rank[page] = ((1 - damping_factor) / num_pages) + \
                                  (damping_factor * sum(page_rank[i] / len(corpus[i]) for i in corpus[page]))

            # Track the maximum change in PageRank values
            max_change = max(max_change, abs(new_page_rank[page] - page_rank[page]))

        # If the maximum change is less than 0.001, break the loop
        if max_change < 0.001:
            break

        # Update the PageRank values for the next iteration
        page_rank = new_page_rank.copy()

    return page_rank

if __name__ == "__main__":
    main()
