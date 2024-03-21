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


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    # Initialize an equal probability distribution dictionary
    pd_dict = {key: 1 / len(corpus) for key in corpus.keys()}

    # Find all linked pages to the current page
    page_links = corpus[page]

    # If no links, return a dictionary with equal probability for all pages
    if len(page_links) == 0:
        return pd_dict

    # Calculate the probability for each linked page on the current page
    links_probability = damping_factor / len(page_links)

    # Calculate the probability for all non-linked pages in the corpus
    damping_probability = (1 - damping_factor) / len(corpus)

    # Update the probability distribution in the dictionary
    for page in pd_dict:
        if page in page_links:
            pd_dict[page] = links_probability + damping_probability
        else: 
            pd_dict[page] = damping_probability

    # Normalize probability
    total_probability = sum(pd_dict.values())
    pd_dict = {page: prob / total_probability for page, prob in pd_dict.items()}
    
    # Check for a total probality sum of 1
    if sum(pd_dict.values()) != 1:
        print(sum(pd_dict.values()), pd_dict.values())
        raise Exception(f"The sum of the probability values of the transition model does not equal 1.\nThe sum is {sum(pd_dict.values())}")
    
    return pd_dict



def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Pick a random first sample
    start_page = random.choice(list(corpus.keys()))

    # Initialize pagerank
    page_rank = {key: 0 for key in corpus.keys()}

    for i in range(n):
        # Get the distribution for the current page
        current_state = transition_model(corpus, start_page, damping_factor)
        # Choose a random new page with weights
        next_page = random.choices(list(current_state.keys()), weights=current_state.values(), k=1)
        # Select new page and add a visited count to pagerank
        start_page = next_page[0]
        page_rank[start_page] += 1
    
    # Calculate pagerank
    for page in page_rank:
        if page_rank[page] > 0:
            page_rank[page] = page_rank[page] / n

    # Normalize probability
    total_probability = sum(page_rank.values())
    page_rank = {page: prob / total_probability for page, prob in page_rank.items()}
    
    # Check for a total probality sum of 1
    if sum(page_rank.values()) != 1:
        print(page_rank.items(),"sum = ", sum(page_rank.values()))
        raise Exception(f"The sum of the probability values of the transition model does not equal 1.\nThe sum is {sum(page_rank.values())}")

    return page_rank





def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initialize page rank with an equal probability distribution
    page_rank = {key: 1 / len(corpus) for key in corpus}

    # Set convergence to false
    convergence = False

    # Initialize threshold
    threshold = 0.001

    # First condition for the probability distribution of a given page
    first_condition = (1 - damping_factor) / len(page_rank)
    
    # Iterate until no change in distribution outside of threshold
    while not convergence:
        # Keep track of current (soon to be previous) page rank
        previous_page_rank = page_rank.copy()

        # Calculate probability of every page in corpus
        for page in page_rank:
            # Initialize second condition, assuming no links
            second_condition = 0
            for p in corpus:
                if page in corpus[p]:
                    # If a page links to the current page, add to second condtion
                    second_condition += previous_page_rank[p] / len(corpus[p])
            # Calculate second condition
            second_condition = damping_factor * second_condition
            # Update page rank
            page_rank[page] = first_condition + second_condition

        # Check for convergence
        count = 0
        for page in page_rank:
            if abs(previous_page_rank[page] - page_rank[page]) > threshold:
                count += 1
        if count == 0:
            convergence = True
    
    # Normalize probability
    total_probability = sum(page_rank.values())
    page_rank = {page: prob / total_probability for page, prob in page_rank.items()}

    # Check for a total probality sum of 1
    if sum(page_rank.values()) != 1:
        print(page_rank.items(),"sum = ", sum(page_rank.values()))
        raise Exception(f"The sum of the probability values of the transition model does not equal 1.\nThe sum is {sum(page_rank.values())}")

    return page_rank





if __name__ == "__main__":
    main()
