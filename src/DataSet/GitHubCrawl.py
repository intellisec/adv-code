import requests
import logging
import argparse
import json
import time
import os
from collections import OrderedDict
from git import Repo  # for cloning repos
import shutil  # for moving files
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

QUERY_URL = 'https://api.github.com/search/repositories?q=language:python&sort=stars&order=desc'
headers = {'Accept': 'application/vnd.github.v3+json'}

REPORT_INTERVAL = 1000


"""
This script is used to crawl the GitHub API for Python repositories
and get the Python files from those repositories.

The script can be run from the command line with the following arguments:
    -n, --number: number of repositories to download
    -d, --code_dir: directory to store the python code
    -c, --cache_dir: intermediate directory to store the cloned repositories (optional, will be created in code_dir if not present)
    -o, --output_metadata: json file to store the list of repositories
    -l, --repo_list: json file containing the repositories to download

Examples:
    Download metadata of the top 1000 repositories, store metadata in repos.json
    and download the python files to the code directory:
    python GitHubCrawl.py -n 1000 -d code -c /tmp/cache -o repos.json

    Download only metadata without cloning the code:
    python GitHubCrawl.py -n 1000 -o repos.json

    Use existing metadata to download the code without querying the GitHub API:
    python GitHubCrawl.py -l repos.json -d code -c /tmp/cache
"""


def getRateLimit():
    # get the rate limit for the GitHub API and the time of the next reset
    # returns a tuple of (limit, remaining, reset_time)
    response = requests.get('https://api.github.com/rate_limit')
    if response.status_code == 200:
        data = response.json()
        limit = data['rate']['limit']
        remaining = data['rate']['remaining']
        reset_time = data['rate']['reset']
        # reset time is in UTC epoch seconds
        # convert to local time
        return (limit, remaining, reset_time)
    else:
        logger.error(f'Error: {response.status_code}')
        return (0, 0, 0)


def loadRepos(jsonfile: str) -> dict:
    # Load repositories from a json file
    # Returns a dictionary of repositories
    # param jsonfile: Name of the json file containing the repositories
    assert os.path.exists(jsonfile), f'{jsonfile} does not exist'
    with open(jsonfile, 'r') as f:
        repos = json.load(f)
    logger.info(f'Loaded {len(repos)} repositories from {jsonfile}')
    return repos


def queryRepos(top_N: int, breakatRateLimit: bool = False) -> dict:
    # Get the top N Python repositories repositories from GitHub
    # Returns a dictionary of repositories
    # param top_N: Number of repositories to download (at least)

    ratelimit, remaining, resetTime = getRateLimit()
    assert ratelimit > 0, 'Rate limit is 0'
    localtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(resetTime))
    logger.log(logging.INFO, f'Rate limit: {remaining} of {ratelimit} remaining. Reset time: {localtime}')
    repos = OrderedDict()
    nexturl = QUERY_URL
    while len(repos) < top_N and nexturl:
        while remaining <= 0:
            logger.warning(f'Rate limit exceeded at {nexturl}. Waiting for reset at {localtime}')
            if breakatRateLimit:
                logger.warning(f'Rate limit exceeded. Breaking at {len(repos)} repositories')
                return repos
            logger.log(logging.INFO, f'Waiting for rate limit reset at {localtime}')
            time.sleep(min(resetTime - time.time(), 5))
            ratelimit, remaining, resetTime = getRateLimit()
            localtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(resetTime))

        response = requests.get(nexturl, headers=headers)
        remaining -= 1
        if response.status_code == 200:
            data = response.json()
            for repo in data['items']:
                repos[repo['full_name']] = repo
                if (len(repos) % REPORT_INTERVAL) == 0:
                    logger.info(f'{len(repos)} repositories queried')
            if 'next' in response.links.keys():
                nexturl = response.links['next']['url']
            else:
                logger.warning('No more repositories to query')
                nexturl = None
        if response.status_code != 200:
            logger.error(f'Error for {nexturl}: {response.status_code}: {response.reason}')
            break
    logger.info(f'Received list of {len(repos)} from GitHub API')
    return repos


def cloneRepos(repos: dict, code_dir: str, cache_dir: Optional[str] = None):
    # Clone the repositories in the repos dictionary, but keep only the .py files
    # param code_dir: Directory to store python code
    # param cache_dir: Directory to store intermediate cloned repositories
    # param repos: Dictionary of repositories to clone
    assert not os.path.exists(code_dir) or (os.path.isdir(code_dir)), f'{code_dir} is not a directory'
    root_dir = os.path.abspath(code_dir)
    os.makedirs(root_dir, exist_ok=True)
    if not cache_dir:
        cache_dir = os.path.join(root_dir, 'cache')
    else:
        cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    cloned = 0
    for full_name, repo in repos.items():
        if cloned > 0 and (cloned % (REPORT_INTERVAL // 10)) == 0:
            logger.info(f'{cloned}/{len(repos)} repositories cloned so far')
        logger.info(f'Cloning {full_name}')
        repo_url = repo['clone_url']

        # is the assumption correct that full_name will never contain dangerous characters?
        Repo.clone_from(repo_url, os.path.join(cache_dir, full_name), depth=1)
        cloned += 1
        # create path and all aprent dirs in root_dir
        # then move the repo from cache_dir to root_dir
        os.makedirs(os.path.join(root_dir, full_name), exist_ok=True)

        # move all .py files from cache_dir to root_dir
        # then delete the cache_dir
        pathPrefix = os.path.join(cache_dir, full_name)
        for root, _, files in os.walk(pathPrefix):
            for file in files:
                if file.endswith('.py'):
                    project_relative = os.path.relpath(root, start=pathPrefix)
                    destdir = os.path.join(root_dir, full_name, project_relative)
                    # make sure destdir is beneath root_dir
                    assert os.path.commonpath([root_dir, destdir]) == root_dir, f'{destdir} is not beneath {root_dir}'
                    os.makedirs(destdir, exist_ok=True)
                    shutil.move(os.path.join(root, file), os.path.join(destdir, file))
        shutil.rmtree(os.path.join(cache_dir, os.path.split(full_name)[0]))
    logger.info(f'Cloned {cloned}/{len(repos)} repositories')


def filterMetaData(repos: dict) -> dict:
    # remove unrequired metadata from the repos dictionary
    # param repos: Dictionary of repositories
    # returns: Dictionary of repositories with unrequired metadata removed
    REQUIRED_KEYS = ['full_name', 'clone_url', 'stargazers_count', 'watchers_count',
                     'forks_count', 'created_at', 'updated_at', 'topics', 'license']
    filtered = OrderedDict()
    for full_name, repo in repos.items():
        filtered[full_name] = {key: repo[key] for key in REQUIRED_KEYS}
    return filtered


def main():
    parser = argparse.ArgumentParser(description='Download top Python repositories from GitHub')
    # either top_n or repo_list must be specified
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-n', '--top_n', type=int, help='Number of repositories to download')
    group.add_argument('-l', '--repo_list', type=str, help='Json file containing list of repositories (will not query GitHub API)')
    parser.add_argument('--log_file', type=str, help='Log file name')
    parser.add_argument('-o', '--output_metadata', type=str, help='Output json file to store repo metadata. If not specified, will not store metadata')
    parser.add_argument('-d', '--code_dir', type=str, help='Directory to store python code')
    parser.add_argument('-c', '--cache_dir', type=str, help='Directory to store cached repositories')
    parser.add_argument('--small_metadata', action='store_true', help='Only store a small subset of metadata')
    args = parser.parse_args()
    if (args.log_file):
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        fh = logging.FileHandler(args.log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.info(f'Downloading top {args.top_n} Python repositories from GitHub')
    if (args.top_n):
        repos = queryRepos(args.top_n)
    else:
        assert args.repo_list, 'repo_list must be specified if top_n is not'
        assert os.path.exists(args.repo_list), f'{args.repo_list} does not exist'
        repos = loadRepos(args.repo_list)
    if args.output_metadata and not args.repo_list:
        if args.small_metadata:
            logger.info('Stripped metadata')
            repos = filterMetaData(repos)
        with open(args.output_metadata, 'w') as f:
            json.dump(repos, f, indent=2)
    if args.code_dir:
        os.makedirs(args.code_dir, exist_ok=True)
        logger.info('Cloning repositories')
        cloneRepos(repos, args.code_dir, cache_dir=args.cache_dir)
    logger.info('Done')


if __name__ == '__main__':
    main()
