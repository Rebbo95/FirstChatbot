import json


def append_json_data(block_of_text):
    intents = []
    lines = block_of_text.splitlines()

    current_intent = None

    for line in lines:
        line = line.strip()
        if line:
            if current_intent is None:
                current_intent = {"tag": line, "patterns": [], "responses": []}
            else:
                current_intent["patterns"].append(line)
                response_line = lines.pop(0).strip()
                current_intent["responses"].append(response_line)

    # Append the last intent to the list
    if current_intent is not None:
        intents.append(current_intent)

    return {"intents": intents}


def main():
    block_of_text = '''
              There it is.
              Just as well we'll never find out.
                Drive!
                Don't look at me, look at the road. That's how accidents happen.
                What's your name? - Cathcart, Robert A.
                What's in the back, Robert A.? - M  s
                OK, Robert, get out of the truck. - l don't want anything from you.- Get!
                Turn it up.
                What's your story, Steamboat?
                Someone cheered too soon. That Rambo? He's on the loose again.
                Will, it's Rambo! He's still alive!
                Jesus Christ! - Get out. Go on!
                What's going on here?
                Watch out! - They're all gonna blow!
                Attention, all civilians!
                For your own safety please clear the streets.
                Stay in your houses and await instructions. l repeat:
                This is a police emergency. Clear the streets at once!
                They found Rambo's body.
                lt stole a truck and blew up a gas station.
                The boy is tough.
                Forget about it. - To hell with your advice!
                Before, you knew he was still alive, didn't you?
                l suspected it.
                Sure, that's why you stayed here. You trained him.
                You taught him how to get out of a cave like that.
                But he won't get out of this place.
                You and your men were never a match for him. So what's changed?
                God knows what damage he'll do.
                You're going to die, Teasle.
                Everybody dies!
                Only one of us has a chance and not because l'm better than him.
                He trusts me.
                l'm his only family, that gives me an advantage.
                What sort of people are you?
                lt's my job, Trautman, it's my town!
                l'm not giving it up to you, Rambo or anyone else!
                Keep out of my way!
                Go nearer! - Don't go,
                lt's too hot!
                Will, Lester here, d'you hear me?
                Come, let's go! Will, Lester here, d'you hear me?
                We've got serious problems. The highway is cut off.
                The truck is here, but no body.
                lt's burnt out. Preston, send the people away!
                Push them back to the south side.
                We don't know how many gas tanks there are under the pumps...
                Go on, go!
                You crazy asshole! - Rambo!
                Rambo, don't do it!
                Listen to me!
                You've got no chance.
                Put your gun down.
                A chopper will fly you to Bragg.
                Cease fire!
                - Yeah? - Cease fire!
                Think what you're doing.
                The building is surrounded. There's no way out.
                There are     men outside with M  s!
                You helped cause this private war.
                You've done enough damage! The mission is over, understood?
                The mission is over!
                Look at them outside.
                Look at them!
                End it, or they'll kill you.
                Do you want that?
                lt's over, Johnny. lt's over!
                Nothing is over! Nothing! You can't just switch it off!
                lt wasn't my war. You asked me, l didn't ask you!
                l did everything to win, but someone didn't let us win.
                And at home at the airport those maggots were protesting.
                They spat at me, called me a baby murderer and shit like that!
                Why protest against me, when they weren't there, didn't experience it?
                lt was hard, but it's in the past.
                For you! Civilian life means nothing to me. There we had a code of honor.
                You watch my back, l watch yours. Here there's nothing!
                You're the last of an elite troop, don't end it like this.
                There l flew helicopters, drove tanks, had equipment worth millions.
                Here l can't even work parking!
                Where is everybody?
                l had a friend who was there for us.
                There were all these guys. There were all these great guys!
                My friends!
                Here there's nothing! D'you remember Dan Forest?
                He wore a black headband. He had found magic markers,
                That he sent to Las Vegas, because we'd always talked about that.
                About the    Chevy Convertible we wanted to drive until the tires fell off.
                ln one of these barns a kid came to us with a kind of shoe cleaning box.
                ''Shine?''
                He kept on asking. Joe said yes.
                l went to get a couple of beers. The box was wired. He opened it...
                There were body parts flying everywhere.
                He lay there and screamed... l have all these pieces of him on me!
                Just like that. l try to get him off me, my friend! l'm covered with him!
                Blood everywhere and so... l try to hold him together,
                But the entrails keep coming out!
                And nobody would help!
                He just said: ''l want to go home!'' And called my name.
                ''l want to go home, Johnny! l want to drive my Chevy!
                But l couldn't find his legs.
                ''l can't find your legs!''
                l can't get it out of my head. lt's seven years ago.
                l see it every day.
                Sometimes l wake up and don't know where l am. l don't talk to anyone.
                Sometimes all day long.
                Sometimes a week.
                l can't get it out of my head.



'''
    

    data = append_json_data(block_of_text)
    with open('../output_data.json', 'w') as file:
        json.dump(data, file, indent=2)
        print("Done")


if __name__ == "__main__":
    main()