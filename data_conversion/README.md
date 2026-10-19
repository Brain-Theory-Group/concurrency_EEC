# Data conversion (`revision_code/data_conversion/`)

The analysis scripts read every dataset in one standard format: a folder
with `edges_list.txt` and `edges_timings.txt` (see `../README.md`, Section 2).
The five scripts here produced that format for the five pairwise temporal
networks (the first five rows of Table 1) from the files as downloaded from
their original sources, which are kept in `data/raw/`.

The converted folders are already in `data/networks/`, so these scripts do
not need to be rerun. They are kept so that the conversion can be checked
and repeated. They are unchanged since the first submission of the paper
and are idempotent: rerunning one produces the same files.

| Script | Reads (in `data/raw/`) | Writes (in `data/networks/`) |
|---|---|---|
| `convert_college_msg.py` | `CollegeMsg.txt` | `college-msg/` |
| `convert_copenhagen_calls.py` | `copenhagen-calls/calls.csv` | `copenhagen-calls/` |
| `convert_friends_family_call.py` | `toSend_realityCommons/FriendsFamily/CallLog.csv` | `friends-family-call/` |
| `convert_ia_reality_call.py` | `ia-reality-call/ia-reality-call.edges` | `ia-reality-call/` |
| `convert_social_evolution_call.py` | `toSend_realityCommons/SocialEvolution/Calls.csv` | `social-evolution-call/` |

Each script

1. reads the raw file;
2. keeps only the records whose two endpoints are both study participants
   (it drops calls to external phone numbers and anonymous identifiers);
3. drops self-loops;
4. makes each record undirected by writing its two endpoints in increasing
   order, because phone calls and one-to-one online messages are
   conventionally treated as undirected interactions (Appendix A of the
   main text);
5. shifts the time stamps so that the earliest event is at t = 0;
6. writes one row per edge to `edges_list.txt` and `edges_timings.txt`, and a
   short provenance note with basic counts to `info.txt`.

The 17 temporal hypergraphs in `data/hypergraphs/` come from the collection
of Benson et al. (PNAS 115, E11221, 2018). Their standard-format files sit in
each folder next to the original files of that collection; no script here
converts them.

Run a script from the project folder (the folder that contains `data/` and
`revision_code/`), e.g.

```bash
python3 revision_code/data_conversion/convert_college_msg.py
```
