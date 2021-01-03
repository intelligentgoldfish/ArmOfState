# ArmOfState V.0.5
Source code for Arm of State Discord bot.

Currently in cloud server deployment.

We are aware that the current GRU model is heavily flawed due to undetected data corruption, and thus has wildly erratic performance.  It is scheduled to be replaced by a two-model ensemble: a BERT-based binary classifier, and a retrained GRU multi-class classifier.

# Purpose
To assist server creators and administrators with server moderation by tracking user interactions.

This goal is accomplished by analyzing individual users' messages and determining one of two things: is their behavior toxic, and if so, how much so?

Over time, users accumulate scores, and the toxicity rankings can be called at any time by any user, from any channel.  Admins may choose to address users with high scores.

Future updates are planned to include the following: remove extremely toxic users to quarantine channels, set server admins as master users for server, generate lists of toxic examples, train on data from server to update self and adapt to users, identify explicit harassment and address it, add additional user commands.

At the moment, we don't guarantee that we'll add any specific feature.  Once the existing structure has been deployed and proven, we'll talk.  Until then, we promise nothing.
