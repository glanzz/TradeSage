import feedparser
import random


class FeedParser:
    def __init__(self):
      self.rss_url = "https://www.ft.com/rss/home"
      self.feed = feedparser.parse(self.rss_url)

    def get_random_feeds(self, flan):
        i = 3
        chosen = []
        while(i>0):
            c = random.choice(self.feed["entries"])
            if c in chosen:
                continue
            chosen.append(flan.generate_question(c["title"]))
            i-=1
        return chosen

