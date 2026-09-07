"""Public-data discovery: searches GitHub repos for candidate datasets, records every source examined."""
import json,time,sys,subprocess,urllib.request
QUERIES={"mempool_fee":["bitcoin mempool historical data","bitcoin fee history csv","bitcoin transaction fees dataset"],
 "hashrate_pool":["bitcoin hashrate history csv","bitcoin mining pool statistics dataset","bitcoin mining pools blocks csv"],
 "hashprice_econ":["hashprice history","bitcoin mining profitability dataset","bitcoin mining energy dataset"],
 "asic":["asic miner specifications dataset","bitcoin mining hardware list json"],
 "blockdata":["bitcoin blocks csv dataset","bitcoin blockchain dataset kaggle csv"]}
out=[]
for fam,qs in QUERIES.items():
    for q in qs:
        url="https://api.github.com/search/repositories?q="+q.replace(' ','+')+"&sort=stars&per_page=8"
        try:
            d=json.load(urllib.request.urlopen(url,timeout=20))
            for r in d.get('items',[]):
                out.append({"family":fam,"query":q,"repo":r['full_name'],"stars":r['stargazers_count'],"size_kb":r['size'],"pushed":r.get('pushed_at','')[:10],"desc":(r.get('description') or '')[:100],"license":(r.get('license') or {}).get('spdx_id')})
            print(f"[{fam}] {q}: {len(d.get('items',[]))} results")
        except Exception as e: print(f"[{fam}] {q}: ERROR {e}")
        time.sleep(7)
json.dump(out,open('research_state/data_sources_examined.json','w'),indent=1)
for r in sorted(out,key=lambda r:-r['stars'])[:40]: print(f"{r['family']:14s} {r['repo']:45s} ★{r['stars']:<5} {r['size_kb']:>8}KB {r['pushed']} {r['license']} - {r['desc'][:60]}")
