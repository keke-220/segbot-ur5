#const self = robot.
#program step(n).

% action load
holds(inhand(self, O),n) :- load(O, n).
:- load(O, n), not holds(at(self, src), n-1).
-holds(unloadedto(O, T), n) :- load(O, n), tableloc(T).

% action unload
-holds(inhand(self, O), n) :- unload(O,T, n).
holds(unloadedto(O,T), n) :- unload(O,T,n), nextto(L,T), holds(at(self, L), n-1).
:- unload(O, T, n), not holds(inhand(self, O), n-1).


% action navigate_to
holds(at(self, Lg), n) :- navigate_to(Ls, Lg, n), holds(at(self, Ls), n-1).
-holds(at(self, Ls), n) :- navigate_to(Ls, Lg, n), holds(at(self, Ls), n-1).
:- navigate_to(Ls, Lg, n), holds(at(self, Lg), n-1).

% static laws
-holds(at(self, L1), n) :- holds(at(self, L2), n), holds(at(self, L1), n-1), L1 != L2.


holds(at(self, L), n) :- holds(at(self, L), n-1), not -holds(at(self, L), n).
holds(unloadedto(O, T), n) :- holds(unloadedto(O, T), n-1), not -holds(unloadedto(O, T), n).
holds(inhand(self, O), n) :- holds(inhand(self, O), n - 1), not -holds(inhand(self, O), n).
