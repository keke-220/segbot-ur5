#include <incmode>.

#program base.

location(src).
location(linit).
tableloc(tinit).
tableloc(t0).
tableloc(t1).
tableloc(t2).

location(l0_top).
location(l0_bot).

location(l1_top).
location(l1_bot).

location(l2_top).
location(l2_bot).

nextto(l0_top, t0).
nextto(l0_bot, t0).
nextto(l1_top, t1).
nextto(l1_bot, t1).
nextto(l2_top, t2).
nextto(l2_bot, t2).
nextto(src, tinit).

object(o0).
object(o1).
object(o2).

% initial state:
init(at(self, linit)).
init(unloadedto(o0,tinit)).
init(unloadedto(o1,tinit)).
init(unloadedto(o2,tinit)).

holds(F,0) :- init(F).

% goal state:
%goal(inhand(self, o0)).
%goal(unloadedto(o0, t0)).
%goal(-inhand(self, o0)).
%goal(at(self, l0_top)).
goal(unloadedto(o0,t0)).
goal(unloadedto(o1,t1)).
goal(unloadedto(o2,t2)).

#program check(n).
% Test
:- query(n), goal(F), not holds(F,n).


