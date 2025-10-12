# Background

A single individual may attempt to set up multiple user accounts with Taptap Send. This could
be an indication that the individual intends to commit fraud, for example by using the same
promo code multiple times or by attempting to circumvent restrictions we have placed on their
existing account.

To prevent this kind of fraud, we would like to identify, as soon as possible after user account
creation, whether the individual has previously set up a Taptap Send user account. Assume we
have the following tables to assist us.

# Table Lists

**users table** contains one row per user account, with the following columns
- user_id: a unique id associated with this user account
- created_at: the timestamp when the user account was created
- date_of_birth: date of birth of the user
- first_name: first name of the user
- middle_names: middle names of the user
- last_name: last name of the user
- is_blocked: whether the account has been blocked for suspected fraud by Taptap Send

**addresses table** contains one row per address, with the following columns
- address_id: unique id associated with this address (a new address_id is created every time a user adds an address, regardless of whether the address details match an existing address in the table)
- created_at: the timestamp when the address was created
- user_id: the user_id from the users table associated with the address
- street_number: the house number of the address
- street_name: the street name of the address
- locality: the city or suburb of the address
- country: the country of the address
- postcode: the postcode of the address

**emails table** contains one row per email address, with the following columns
- email_id: unique id associated with this email (a new email_id is created every time a user adds an email, regardless of whether the email details match an existing email in the table)
- created_at: the timestamp when the email was created
- user_id: the user_id from the users table associated with the email
- email: the email address

**phones table** contains one row per phone number, with the following columns
- phone_id: unique id associated with this phone number (a new phone_id is created every time a user adds a phone number, regardless of whether the phone number details match an existing phone number in the table)
- created_at: the timestamp when the phone number was created
- user_id: the user_id from the users table associated with the phone
- phone_number: the phone number without the country code
- country_code: the country code of the phone number


# Task

1. Write one or more SQL queries in a dialect of your choice to help you identify cases where a single individual has set up multiple user accounts
2. Describe how you would incorporate these SQL queries into a business process to prevent individuals from using multiple user accounts to commit fraud
3. If you have time, outline how you could use a machine learning or rule-based approach for improved detection